// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.searcher;

import ai.vespa.examples.Centroids;
import ai.vespa.examples.DimensionReducer;
import com.google.inject.Inject;
import com.yahoo.component.chain.dependencies.After;
import com.yahoo.prelude.query.AndItem;
import com.yahoo.prelude.query.DotProductItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.WeightedSetItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import java.time.Duration;
import java.util.List;
import java.util.Optional;

@After({"ExternalYql","ClipEmbedding"})
public class SPANNSearcher extends Searcher {

    private final Centroids clustering;
    private final DimensionReducer reducer;
    private final static String centroidField = "centroids";

    @Inject
    public SPANNSearcher(Centroids clusteringComponent, DimensionReducer reducer) {
        this.clustering = clusteringComponent;
        this.reducer = reducer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        Optional<Tensor> optionalTensor = query.getRanking().getFeatures().getTensor("query(q)");
        if (optionalTensor.isEmpty())
            return execution.search(query);
        Tensor reducedQueryTensor = reducer.reduce(optionalTensor.get());
        query.getRanking().getFeatures().put("query(q_reduced)", reducedQueryTensor);

        Centroids.CentroidResult centroidResult = clustering.getCentroids(
                reducedQueryTensor,
                query.properties().getInteger("spann.clusters", 64),
                query.properties().getInteger("spann.additional-clusters", 128),
                Duration.ofMillis(query.getTimeLeft()),
                execution
        );
        query.trace("Got top k centroids " + centroidResult.getCentroids(), 3);
        if(centroidResult.getCentroidResult().hits().getError() != null) {
            return new Result(query, centroidResult.getCentroidResult().hits().getError());
        }
        List<Centroids.Centroid> retainedCentroids = clustering.prune(centroidResult.getCentroids(),
            query.properties().getDouble("spann.distance-prune-threshold",0.8));

       buildQuery(query, retainedCentroids);
       query.getModel().setSources("if");
       return execution.search(query);
    }

    void buildQuery(Query query, List<Centroids.Centroid> centroids) {
        WeightedSetItem setItem = new WeightedSetItem(centroidField);
        setItem.setFilter(true);
        setItem.setRanked(false);
        centroids.forEach(c -> setItem.addToken(c.getId(),1));
        Item root = query.getModel().getQueryTree().getRoot();
        if(root instanceof AndItem)
            ((AndItem) root).addItem(setItem);
        else  {
            AndItem andItem = new AndItem();
            andItem.addItem(root);
            andItem.addItem(setItem);
            query.getModel().getQueryTree().setRoot(andItem);
        }
        //Allow termwise query evaluation of large OR-like search
        query.getRanking().getMatching().setTermwiselimit(0);
    }
}
