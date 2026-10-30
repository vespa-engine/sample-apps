// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.colbert;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link HierarchicalTokenPooling}.
 * <p>
 * Covers: pdistCosine, linkageWard, fclusterMaxclust, poolTokens correctness
 * and performance for realistic ColBERT document sizes.
 */
class HierarchicalTokenPoolingTest {

    private static final int DIM = 128;

    /** Generate L2-normalised random embeddings in row-major flat array. */
    private static double[] randomEmbeddings(int n, int dim, long seed) {
        Random rng = new Random(seed);
        double[] emb = new double[n * dim];
        for (int i = 0; i < n; i++) {
            double norm = 0;
            for (int d = 0; d < dim; d++) {
                emb[i * dim + d] = rng.nextGaussian();
                norm += emb[i * dim + d] * emb[i * dim + d];
            }
            norm = Math.sqrt(norm);
            for (int d = 0; d < dim; d++) emb[i * dim + d] /= norm;
        }
        return emb;
    }

    // ---------------------------------------------------------------
    // pdistCosine
    // ---------------------------------------------------------------

    @Test
    void pdistCosine_identicalVectors() {
        double[] emb = randomEmbeddings(1, 4, 42);
        double[] triple = new double[3 * 4];
        System.arraycopy(emb, 0, triple, 0, 4);
        System.arraycopy(emb, 0, triple, 4, 4);
        System.arraycopy(emb, 0, triple, 8, 4);

        double[] dists = HierarchicalTokenPooling.pdistCosine(triple, 3, 4);
        assertEquals(3, dists.length);
        for (double d : dists) assertEquals(0.0, d, 1e-10);
    }

    @Test
    void pdistCosine_orthogonalVectors() {
        // Three orthogonal unit vectors
        double[] emb = new double[3 * 3];
        emb[0] = 1; emb[4] = 1; emb[8] = 1;
        double[] dists = HierarchicalTokenPooling.pdistCosine(emb, 3, 3);
        assertEquals(3, dists.length);
        for (double d : dists) assertEquals(1.0, d, 1e-10);
    }

    @Test
    void pdistCosine_outputLength() {
        int n = 20;
        double[] emb = randomEmbeddings(n, DIM, 42);
        double[] dists = HierarchicalTokenPooling.pdistCosine(emb, n, DIM);
        assertEquals(n * (n - 1) / 2, dists.length);
        for (double d : dists) {
            assertTrue(d >= 0.0 && d <= 2.0, "Distance out of range: " + d);
        }
    }

    // ---------------------------------------------------------------
    // linkageWard
    // ---------------------------------------------------------------

    @Test
    void linkageWard_threePoints() {
        // Two close points + one far point
        double[] emb = {1.0, 0.0,   0.9, 0.1,   0.0, 1.0};
        // Normalise
        for (int i = 0; i < 3; i++) {
            double n = Math.sqrt(emb[i*2]*emb[i*2] + emb[i*2+1]*emb[i*2+1]);
            emb[i*2] /= n; emb[i*2+1] /= n;
        }
        double[] dists = HierarchicalTokenPooling.pdistCosine(emb, 3, 2);
        double[][] Z = HierarchicalTokenPooling.linkageWard(dists, 3);

        assertEquals(2, Z.length);
        // First merge: the two closest (0 and 1)
        assertTrue((int) Z[0][0] == 0 || (int) Z[0][0] == 1);
        assertTrue((int) Z[0][1] == 0 || (int) Z[0][1] == 1);
        assertEquals(2, (int) Z[0][3]);
        // All distances non-negative
        for (double[] row : Z) assertTrue(row[2] >= 0);
    }

    @Test
    void linkageWard_clusterSizes() {
        int n = 30;
        double[] emb = randomEmbeddings(n, DIM, 123);
        double[] dists = HierarchicalTokenPooling.pdistCosine(emb, n, DIM);
        double[][] Z = HierarchicalTokenPooling.linkageWard(dists, n);

        assertEquals(n - 1, Z.length);
        for (double[] row : Z) assertTrue(row[3] >= 2, "Merge size must be >= 2");
        assertEquals(n, (int) Z[n - 2][3], "Final merge must contain all observations");
    }

    // ---------------------------------------------------------------
    // fclusterMaxclust
    // ---------------------------------------------------------------

    @Test
    void fcluster_correctNumberOfClusters() {
        int n = 50;
        double[] emb = randomEmbeddings(n, DIM, 42);
        double[] dists = HierarchicalTokenPooling.pdistCosine(emb, n, DIM);
        double[][] Z = HierarchicalTokenPooling.linkageWard(dists, n);

        for (int k : new int[]{1, 2, 5, 10, 25, 50}) {
            int[] labels = HierarchicalTokenPooling.fclusterMaxclust(Z, n, k);
            assertEquals(n, labels.length);
            int maxLabel = 0;
            for (int l : labels) maxLabel = Math.max(maxLabel, l);
            assertEquals(k, maxLabel, "Expected " + k + " clusters, got " + maxLabel);
            for (int l : labels) assertTrue(l >= 1, "Labels must be 1-indexed");
        }
    }

    // ---------------------------------------------------------------
    // poolTokens
    // ---------------------------------------------------------------

    @Test
    void poolTokens_outputShape() {
        int nTokens = 50;
        double[] emb = randomEmbeddings(nTokens, DIM, 42);
        double[] pooled = HierarchicalTokenPooling.poolTokens(emb, nTokens, DIM, 2, true);
        // CLS (1) + ceil(49/2) = 1 + 25 = 26
        assertEquals(26 * DIM, pooled.length);
    }

    @Test
    void poolTokens_clsPreserved() {
        int nTokens = 20;
        double[] emb = randomEmbeddings(nTokens, DIM, 42);
        double[] pooled = HierarchicalTokenPooling.poolTokens(emb, nTokens, DIM, 2, true);
        for (int d = 0; d < DIM; d++) {
            assertEquals(emb[d], pooled[d], 1e-12, "CLS token dimension " + d + " not preserved");
        }
    }

    @Test
    void poolTokens_normalised() {
        int nTokens = 40;
        double[] emb = randomEmbeddings(nTokens, DIM, 42);
        double[] pooled = HierarchicalTokenPooling.poolTokens(emb, nTokens, DIM, 2, true);
        int nPooled = pooled.length / DIM;
        for (int t = 0; t < nPooled; t++) {
            double norm = 0;
            for (int d = 0; d < DIM; d++) {
                norm += pooled[t * DIM + d] * pooled[t * DIM + d];
            }
            assertEquals(1.0, Math.sqrt(norm), 1e-8, "Token " + t + " not normalised");
        }
    }

    @Test
    void poolTokens_poolFactor3() {
        int nTokens = 31;
        double[] emb = randomEmbeddings(nTokens, DIM, 42);
        double[] pooled = HierarchicalTokenPooling.poolTokens(emb, nTokens, DIM, 3, true);
        // CLS (1) + ceil(30/3) = 1 + 10 = 11
        assertEquals(11 * DIM, pooled.length);
    }

    @Test
    void poolTokens_noPooling() {
        int nTokens = 20;
        double[] emb = randomEmbeddings(nTokens, DIM, 42);
        double[] pooled = HierarchicalTokenPooling.poolTokens(emb, nTokens, DIM, 0, true);
        assertEquals(nTokens * DIM, pooled.length);
        assertArrayEquals(emb, pooled, 1e-15);
    }

    @Test
    void poolTokens_singleToken() {
        double[] emb = randomEmbeddings(1, DIM, 42);
        double[] pooled = HierarchicalTokenPooling.poolTokens(emb, 1, DIM, 2, false);
        assertArrayEquals(emb, pooled, 1e-15);
    }

    // ---------------------------------------------------------------
    // Performance
    // ---------------------------------------------------------------

    @Test
    void performance_pooling512tokens() {
        double[] emb = randomEmbeddings(512, DIM, 42);
        // Warm-up
        HierarchicalTokenPooling.poolTokens(emb, 512, DIM, 2, true);

        int runs = 3;
        long t0 = System.nanoTime();
        for (int i = 0; i < runs; i++) {
            HierarchicalTokenPooling.poolTokens(emb, 512, DIM, 2, true);
        }
        double ms = (System.nanoTime() - t0) / 1e6 / runs;
        System.out.printf("  poolTokens n=512: %.1f ms%n", ms);
        assertTrue(ms < 2000, "Pooling 512 tokens took " + ms + " ms, expected < 2000");
    }

    @Test
    void performance_pooling128tokens() {
        double[] emb = randomEmbeddings(128, DIM, 42);
        HierarchicalTokenPooling.poolTokens(emb, 128, DIM, 2, true);

        int runs = 5;
        long t0 = System.nanoTime();
        for (int i = 0; i < runs; i++) {
            HierarchicalTokenPooling.poolTokens(emb, 128, DIM, 2, true);
        }
        double ms = (System.nanoTime() - t0) / 1e6 / runs;
        System.out.printf("  poolTokens n=128: %.1f ms%n", ms);
        assertTrue(ms < 500, "Pooling 128 tokens took " + ms + " ms, expected < 500");
    }

    @Test
    void performance_pdist512() {
        double[] emb = randomEmbeddings(512, DIM, 42);
        HierarchicalTokenPooling.pdistCosine(emb, 512, DIM);

        int runs = 5;
        long t0 = System.nanoTime();
        for (int i = 0; i < runs; i++) {
            HierarchicalTokenPooling.pdistCosine(emb, 512, DIM);
        }
        double ms = (System.nanoTime() - t0) / 1e6 / runs;
        System.out.printf("  pdistCosine n=512: %.1f ms%n", ms);
        assertTrue(ms < 500, "pdist 512 took " + ms + " ms, expected < 500");
    }
}
