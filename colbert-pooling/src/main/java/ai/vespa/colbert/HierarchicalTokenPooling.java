// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.colbert;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Hierarchical token pooling for ColBERT multi-vector embeddings.
 * <p>
 * Implements Ward's agglomerative clustering using the Nearest-Neighbor Chain
 * (NNC) algorithm in O(n²), matching the behaviour of the Rust reference at
 * https://github.com/lightonai/next-plaid/blob/main/next-plaid-onnx/src/hierarchy.rs
 * <p>
 * The pooling pipeline:
 * <ol>
 *   <li>Compute pairwise cosine distances between token embeddings.</li>
 *   <li>Build a Ward linkage dendrogram via NNC.</li>
 *   <li>Cut the dendrogram to ceil(n / poolFactor) clusters.</li>
 *   <li>Replace each cluster with its L2-normalised centroid.</li>
 *   <li>Preserve the CLS token unchanged.</li>
 * </ol>
 */
public final class HierarchicalTokenPooling {

    private HierarchicalTokenPooling() {}

    // ---------------------------------------------------------------
    // Pairwise cosine distance – condensed upper-triangular form
    // ---------------------------------------------------------------

    /**
     * Compute pairwise cosine distances between rows of a flat row-major embedding matrix.
     *
     * @param embeddings row-major float array of shape (n, dim)
     * @param n          number of embeddings (rows)
     * @param dim        embedding dimension
     * @return condensed distance vector of length n*(n-1)/2
     */
    public static double[] pdistCosine(double[] embeddings, int n, int dim) {
        double[] norms = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            int off = i * dim;
            for (int d = 0; d < dim; d++) {
                double v = embeddings[off + d];
                s += v * v;
            }
            norms[i] = Math.sqrt(s);
            if (norms[i] < 1e-10) norms[i] = 1e-10;
        }

        int len = n * (n - 1) / 2;
        double[] dists = new double[len];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            int offI = i * dim;
            for (int j = i + 1; j < n; j++) {
                double dot = 0;
                int offJ = j * dim;
                for (int d = 0; d < dim; d++) {
                    dot += embeddings[offI + d] * embeddings[offJ + d];
                }
                double sim = dot / (norms[i] * norms[j]);
                dists[idx++] = Math.max(0.0, Math.min(2.0, 1.0 - sim));
            }
        }
        return dists;
    }

    // ---------------------------------------------------------------
    // Ward linkage – Nearest-Neighbor Chain algorithm
    // ---------------------------------------------------------------

    /**
     * Ward's method hierarchical clustering using the NNC algorithm.
     *
     * @param dists condensed cosine distance vector from {@link #pdistCosine}
     * @param n     number of observations
     * @return linkage matrix of shape (n-1, 4): [id_a, id_b, distance, size]
     */
    public static double[][] linkageWard(double[] dists, int n) {
        if (n <= 1) return new double[0][4];

        int total = 2 * n - 1;

        // Full symmetric squared-distance matrix for all clusters (original + merged)
        double[][] D = new double[total][total];
        for (double[] row : D) Arrays.fill(row, Double.MAX_VALUE);

        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double d2 = dists[idx] * dists[idx];
                D[i][j] = d2;
                D[j][i] = d2;
                idx++;
            }
        }

        int[] sizes = new int[total];
        Arrays.fill(sizes, 0, n, 1);

        boolean[] active = new boolean[total];
        Arrays.fill(active, 0, n, true);

        double[][] Z = new double[n - 1][4];
        int[] chain = new int[n + 1];
        int chainSize = 0;

        for (int step = 0; step < n - 1; step++) {
            if (chainSize == 0) {
                for (int i = 0; i < total; i++) {
                    if (active[i]) {
                        chain[chainSize++] = i;
                        break;
                    }
                }
            }

            int a, b;
            while (true) {
                int cur = chain[chainSize - 1];
                // Find nearest active neighbour
                double minD = Double.MAX_VALUE;
                int nn = -1;
                for (int k = 0; k < total; k++) {
                    if (k != cur && active[k] && D[cur][k] < minD) {
                        minD = D[cur][k];
                        nn = k;
                    }
                }
                if (chainSize >= 2 && nn == chain[chainSize - 2]) {
                    b = chain[--chainSize];
                    a = chain[--chainSize];
                    break;
                }
                chain[chainSize++] = nn;
            }

            if (a > b) { int t = a; a = b; b = t; }

            double mergeD2 = D[a][b];
            int newId = n + step;
            Z[step][0] = a;
            Z[step][1] = b;
            Z[step][2] = Math.sqrt(Math.max(mergeD2, 0.0));
            Z[step][3] = sizes[a] + sizes[b];
            sizes[newId] = sizes[a] + sizes[b];

            active[a] = false;
            active[b] = false;

            // Ward distance update – vectorised over active clusters
            double na = sizes[a], nb = sizes[b];
            for (int k = 0; k < total; k++) {
                if (active[k]) {
                    double nk = sizes[k];
                    double dNew = ((na + nk) * D[a][k] + (nb + nk) * D[b][k] - nk * mergeD2) / (na + nb + nk);
                    D[newId][k] = dNew;
                    D[k][newId] = dNew;
                }
            }
            active[newId] = true;
        }
        return Z;
    }

    // ---------------------------------------------------------------
    // Flat cluster assignment – maxclust criterion
    // ---------------------------------------------------------------

    private static int find(int[] parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    /**
     * Cut a Ward dendrogram to form exactly {@code maxClusters} flat clusters.
     * <p>
     * Selects the merges with the smallest distances (robust to non-monotonic
     * merge orderings from the NNC algorithm).
     *
     * @param Z           linkage matrix from {@link #linkageWard}
     * @param n           number of original observations
     * @param maxClusters desired number of clusters
     * @return 1-indexed cluster labels of length n
     */
    public static int[] fclusterMaxclust(double[][] Z, int n, int maxClusters) {
        maxClusters = Math.max(1, Math.min(maxClusters, n));
        int nMerges = n - maxClusters;

        if (nMerges == 0) {
            int[] labels = new int[n];
            for (int i = 0; i < n; i++) labels[i] = i + 1;
            return labels;
        }

        // Sort merge indices by ascending distance
        Integer[] sortedIdx = new Integer[Z.length];
        for (int i = 0; i < sortedIdx.length; i++) sortedIdx[i] = i;
        Arrays.sort(sortedIdx, (x, y) -> Double.compare(Z[x][2], Z[y][2]));

        int[] parent = new int[2 * n - 1];
        for (int i = 0; i < parent.length; i++) parent[i] = i;

        for (int m = 0; m < nMerges; m++) {
            int mi = sortedIdx[m];
            int a = (int) Z[mi][0], b = (int) Z[mi][1];
            int newId = n + mi;
            parent[find(parent, a)] = newId;
            parent[find(parent, b)] = newId;
        }

        int[] labels = new int[n];
        Map<Integer, Integer> clusterMap = new HashMap<>();
        int nextLabel = 1;
        for (int i = 0; i < n; i++) {
            int root = find(parent, i);
            Integer label = clusterMap.get(root);
            if (label == null) {
                label = nextLabel++;
                clusterMap.put(root, label);
            }
            labels[i] = label;
        }
        return labels;
    }

    // ---------------------------------------------------------------
    // Token pooling entry-point
    // ---------------------------------------------------------------

    /**
     * Pool ColBERT token embeddings via hierarchical clustering.
     *
     * @param embeddings row-major array of shape (nTokens, dim)
     * @param nTokens    number of tokens
     * @param dim        embedding dimension
     * @param poolFactor reduction factor (2 = halve tokens)
     * @param skipFirst  preserve first token (CLS) unchanged
     * @return row-major array of shape (nPooled, dim) with L2-normalised centroids
     */
    public static double[] poolTokens(double[] embeddings, int nTokens, int dim,
                                       int poolFactor, boolean skipFirst) {
        if (poolFactor <= 1 || nTokens <= 1) {
            return Arrays.copyOf(embeddings, nTokens * dim);
        }

        int startIdx = skipFirst ? 1 : 0;
        int nToPool = nTokens - startIdx;

        if (nToPool <= 1) {
            return Arrays.copyOf(embeddings, nTokens * dim);
        }

        int nClusters = Math.max(1, (int) Math.ceil((double) nToPool / poolFactor));
        if (nClusters >= nToPool) {
            return Arrays.copyOf(embeddings, nTokens * dim);
        }

        // Extract tokens to pool (skip CLS if requested)
        double[] tokens = new double[nToPool * dim];
        System.arraycopy(embeddings, startIdx * dim, tokens, 0, nToPool * dim);

        // Cluster
        double[] dists = pdistCosine(tokens, nToPool, dim);
        double[][] Z = linkageWard(dists, nToPool);
        int[] labels = fclusterMaxclust(Z, nToPool, nClusters);

        // Compute centroids
        int nActualClusters = 0;
        for (int l : labels) nActualClusters = Math.max(nActualClusters, l);

        double[] centroids = new double[nActualClusters * dim];
        int[] counts = new int[nActualClusters];
        for (int i = 0; i < nToPool; i++) {
            int c = labels[i] - 1; // 1-indexed to 0-indexed
            counts[c]++;
            int cOff = c * dim;
            int tOff = i * dim;
            for (int d = 0; d < dim; d++) {
                centroids[cOff + d] += tokens[tOff + d];
            }
        }
        for (int c = 0; c < nActualClusters; c++) {
            int cOff = c * dim;
            double norm = 0;
            for (int d = 0; d < dim; d++) {
                centroids[cOff + d] /= counts[c];
                norm += centroids[cOff + d] * centroids[cOff + d];
            }
            norm = Math.sqrt(norm);
            if (norm < 1e-10) norm = 1e-10;
            for (int d = 0; d < dim; d++) {
                centroids[cOff + d] /= norm;
            }
        }

        // Build result: CLS (if skipFirst) + pooled centroids
        int nPooled = (skipFirst ? 1 : 0) + nActualClusters;
        double[] result = new double[nPooled * dim];

        if (skipFirst) {
            System.arraycopy(embeddings, 0, result, 0, dim); // CLS token
        }
        System.arraycopy(centroids, 0, result, (skipFirst ? 1 : 0) * dim, nActualClusters * dim);

        return result;
    }
}
