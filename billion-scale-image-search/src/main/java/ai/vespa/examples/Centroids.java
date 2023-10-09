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

public class Centroids extends AbstractComponent {

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
        private double closeness;
        private Centroid(int id, double closeness) {
            this.id = id;
            this.closeness = closeness;
        }

        public Integer getId() {
            return id;
        }
        public double getCloseness() {
            return closeness;
        }
        @Override
        public String toString() {
            return id  + ", closeness: " + getCloseness();
        }
    }

    /**
     * Get
     * @param tensor The input reduced vector
     * @param k The number of clusters to find
     * @param extraK Additional parameter for improving recall
     * @param execution The execution to use
     * @return A centroid result
     */

    public CentroidResult getCentroids(Tensor tensor, int k, int extraK, Duration timeout, Execution execution) {
        Result result = execution.search(buildNNQuery(k, extraK, tensor, timeout));
        List<Centroid> centroids = new ArrayList<>(k);
        CentroidResult centroidResult = new CentroidResult(centroids,result);
        if(result.getConcreteHitCount() == 0) {
            return centroidResult;
        }
        for(Hit hit: result.hits()){
            FeatureData featureData = (FeatureData)hit.getField("matchfeatures");
            if(featureData == null)
                throw new RuntimeException("No matchfeatures in hit " + hit);
            double closeness = featureData.getDouble("closeness(field,reduced_vector)");
            int id = featureData.getDouble("attribute(id)").intValue();
            centroids.add(new Centroid(id,closeness));
        }
        return centroidResult;
    }

    /**
     * Centroid dynamic pruning using closeness
     *
     * Instead of searching closest K posting lists (centroids) for all queries,
     * we dynamically decide a posting list for a centroid to be searched only if the distance
     * between its centroid and query is almost the same as the distance between query and the closest
     * centroid:
     *
     * @param centroids the retrieved centroids
     * @param factor, range [0,1]. 1.0 only retain the closest, 0 retain all k.
     * @return retained centroids after pruning
     */

    public List<Centroid> prune(List<Centroid> centroids, double factor) {
        List<Centroid> retained = new ArrayList<>();
        if(centroids.isEmpty())
            return retained;
        Iterator<Centroid> iterator = centroids.iterator();
        Centroid closestCentroid = iterator.next();
        retained.add(closestCentroid);
        double minCloseness = closestCentroid.getCloseness()*factor;
        while(iterator.hasNext()) {
            Centroid nextClosestCentroid = iterator.next();
            if (nextClosestCentroid.getCloseness() > minCloseness)
                retained.add(nextClosestCentroid);
        }
        return retained;
    }

    /**
     * Builds the NearestNeighborItem
     * @param k the k nearest neighbors
     * @param extraK - HNSW parameer to improve graph accuracy
     * @param queryVector - The vector used to search the graph
     * @parem timeout - the query timeout
     * @return Query instance with nearest neighbor search item
     */

    private Query buildNNQuery(int k, int extraK, Tensor queryVector, Duration timeout) {
        NearestNeighborItem nn = new NearestNeighborItem("reduced_vector", "q_reduced");
        nn.setAllowApproximate(true);
        nn.setTargetNumHits(k);
        nn.setHnswExploreAdditionalHits(extraK);
        Query query = new Query();
        query.setTimeout(timeout.toMillis());
        query.setHits(k);
        query.getModel().getQueryTree().setRoot(nn);
        query.getRanking().getMatching().setPostFilterThreshold(0);
        query.getRanking().setProfile("default");
        query.getRanking().getFeatures().put("query(q_reduced)",queryVector);
        query.getModel().setSources("graph");
        query.getModel().setRestrict("centroid");
        return query;
    }
}
