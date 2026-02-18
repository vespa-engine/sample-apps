// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.searcher;

import ai.vespa.examples.ClusteringComponent;
import com.google.inject.Inject;
import com.yahoo.prelude.query.DotProductItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;

import java.time.Duration;
import java.util.List;
import java.util.Optional;

public class SPANNSearcher extends Searcher {

    private ClusteringComponent clustering;

    @Inject
    public SPANNSearcher(ClusteringComponent clusteringComponent) {
        this.clustering = clusteringComponent;
    }

    @Override
    public Result search(Query query, Execution execution) {
        Optional<Tensor> optionalTensor = query.getRanking().getFeatures().getTensor("query(q)");
        if (optionalTensor.isEmpty())
            return new Result(query,
                    ErrorMessage.createBadRequest("No query tensor 'query(q)' in input request"));

        int clusters = query.properties().getInteger("spann.clusters", 3);
        int hnswExploreHits = query.properties().getInteger("spann.additional-clusters",clusters*2);
        double pruneThreshold = query.properties().getDouble("spann.distance-prune-threshold",0.8);
        int reRankCount = query.properties().getInteger("spann.rank-count",1000);

        ClusteringComponent.CentroidResult centroidResult = clustering.getCentroids(
                optionalTensor.get(),
                clusters,
                hnswExploreHits,
                Duration.ofSeconds(5),
                execution
        );

        List<ClusteringComponent.Centroid> prunedCentroids = clustering.prune(centroidResult.getCentroids(),
                pruneThreshold);

        DotProductItem dp = new DotProductItem("neighbors");
        for (ClusteringComponent.Centroid c : prunedCentroids) {
            dp.addToken(c.getId(), c.getIntCloseness());
        }
        query.getModel().getQueryTree().setRoot(dp);
        query.getModel().setSources("if");
        query.getRanking().setRerankCount(reRankCount);
        query.getRanking().getMatching().setTermwiselimit(0);
        return mergeResult(execution.search(query), centroidResult);
    }

    private Result mergeResult(Result result, ClusteringComponent.CentroidResult centroidResult) {
        result.getCoverage(false).merge(centroidResult.getCentroidResult().getCoverage(false));
        for(Hit hit: result.hits())  {
            hit.setFilled("id");
            FeatureData featureData = (FeatureData)hit.getField("matchfeatures");
            hit.removeField("matchfeatures");
            Integer id = featureData.getDouble("attribute(id)").intValue();
            hit.setField("id", id);
            hit.setId(id.toString());
        }
        for (ClusteringComponent.Centroid c : centroidResult.getCentroids()) {
            Hit hit = c.getHit();
            hit.setField("id", c.getId());
            hit.setFilled("id");
            hit.setId(c.getId().toString());
            hit.removeField("matchfeatures");
            result.hits().add(hit);
        }
        //re-sort and trim after sort
        result.hits().sort();
        result.hits().trim(0, result.getQuery().getHits());
        return result;
    }
}
