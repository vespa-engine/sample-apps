// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples;

import com.yahoo.component.AbstractComponent;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class ClusteringComponent extends AbstractComponent {

    public static class CentroidResult {

        private final List<Centroid> centroids;
        private final Result result;

        private CentroidResult(List<Centroid> centroids, Result result) {
            this.centroids = centroids;
            this.result = result;
        }
        public boolean isEmpty()  {
            return centroids.isEmpty();
        }
        public List<Centroid> getCentroids() {
            return this.centroids;
        }
        public Result getCentroidResult() {
            return this.result;
        }
    }

    public static class Centroid {
        private final int id;
        private final double closeness;
        private final double distance;
        private final Hit hit;
        private static int INT_SCALE = 100000;

        private Centroid(int id, double closeness, double distance, Hit hit) {
            this.id = id;
            this.closeness = closeness;
            this.distance = distance;
            this.hit = hit;
        }

        public Integer getId() {
            return id;
        }

        public double getCloseness() {
            return closeness;
        }

        public int getIntCloseness() {
            return (int)(INT_SCALE*closeness);
        }

        public double getDistance() {
            return distance;
        }

        public Hit getHit() {
            return this.hit;
        }

        @Override
        public String toString() {
            return id  + ", distance:" + getDistance();
        }
    }

    /**
     * Get
     * @param queryTensor The input vector
     * @param k The number of clusters to find
     * @param extraK Additional parameter for improving recall
     * @param execution The execution to use
     * @return A centroid result
     */

    public CentroidResult getCentroids(Tensor queryTensor, int k, int extraK, Duration timeout, Execution execution) {
        Result result = execution.search(buildNNQuery(k,extraK,queryTensor, timeout));
        List<Centroid> centroids = new ArrayList<>(k);
        CentroidResult centroidResult = new CentroidResult(centroids,result);
        if(result.getConcreteHitCount() == 0) {
            return centroidResult;
        }
        for(Hit hit: result.hits()){
            FeatureData featureData = (FeatureData)hit.getField("matchfeatures");
            if(featureData == null)
                throw new RuntimeException("No matchfeatures in hit " + hit);
            double closeness = featureData.getDouble("closeness(field,vector)");
            double distance = featureData.getDouble("distance(field,vector)");
            int id = featureData.getDouble("attribute(id)").intValue();
            centroids.add(new Centroid(id,closeness,distance,hit));
        }
        return centroidResult;
    }

    /**
     * Query-aware dynamic pruning
     *
     * Instead of searching closest K
     * posting lists (centroids) for all queries, we dynamically decide a posting list to be searched only if the distance
     * between its centroid and query is almost the same as the distance between query and the closest
     * centroid:
     *
     * @param centroids the k nearest centroids
     * @param factor, range [0,1]. 1.0 only retain the closest, 0 retain all k.
     * @return
     */

    public List<Centroid> prune(List<Centroid> centroids, double factor) {
        List<Centroid> pruned = new ArrayList<>();
        if(centroids.isEmpty())
            return pruned;
        Iterator<Centroid> iterator = centroids.iterator();
        Centroid closestCentroid = iterator.next();
        pruned.add(closestCentroid);
        double minCloseness = closestCentroid.getCloseness()*factor;
        while(iterator.hasNext()) {
            Centroid nextClosestCentroid = iterator.next();
            if (nextClosestCentroid.getCloseness() > minCloseness)
                pruned.add(nextClosestCentroid);
        }
        return pruned;
    }

    /**
     *
     * @param k the k nearest neighbors
     * @param extraK - Improve graph accuracy
     * @param queryVector - The document vector used to search the graph
     * @return Query instance with nearest neighbor search item
     */

    private Query buildNNQuery(int k, int extraK, Tensor queryVector, Duration timeout) {
        NearestNeighborItem nn = new NearestNeighborItem("vector", "q");
        nn.setAllowApproximate(true);
        nn.setTargetNumHits(k);
        nn.setHnswExploreAdditionalHits(extraK);
        Query query = new Query();
        query.setTimeout(timeout.toMillis());
        query.setHits(k);
        query.getModel().getQueryTree().setRoot(nn);
        query.getRanking().setProfile("graph");
        query.getRanking().getMatching().setPostFilterThreshold(0);
        query.getRanking().getFeatures().put("query(q)",queryVector);
        query.getModel().setSources("graph");
        return query;
    }




}
