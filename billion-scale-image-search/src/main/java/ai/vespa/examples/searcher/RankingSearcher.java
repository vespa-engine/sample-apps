// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.searcher;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.google.inject.Inject;
import com.yahoo.component.chain.dependencies.After;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.Optional;

@After("ExternalYql")
@Provides("Ranking")
public class RankingSearcher extends Searcher {

    private final ModelsEvaluator modelsEvaluator;

    @Inject
    public RankingSearcher(ModelsEvaluator evaluator) {
        this.modelsEvaluator = evaluator;
    }

    @Override
    public Result search(Query query, Execution execution) {
        int maxRankCount = query.properties().getInteger("rank-count", 1000);
        query.getPresentation().getSummaryFields().add("vector");
        query.setHits(maxRankCount);
        //Execute first protocol phase
        Result result = execution.search(query);
        //Execute fill phase if not done before - gets query.getHits vectors
        ensureFilled(result, "vector-summary", execution);
        Optional<Tensor> q = query.getRanking().getFeatures().getTensor("query(q)");
        if(q.isEmpty()) //No vector query
            return result;
        reScore(result,q.get());
        result.hits().sort();
        return result;
    }

    private void reScore(Result result, Tensor query) {
        int size = result.getConcreteHitCount();
        if (size == 0)
            return;
        TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).
                indexed("d0", size).indexed("d1", 768).build();
        Tensor.Builder batch = Tensor.Builder.of(type);
        int rank = 0;
        for (Hit h : result.hits()) {
            IndexedTensor vector = (IndexedTensor) h.getField("vector");
            for (int i = 0; i < vector.size(); i++)
                batch.cell(vector.get(i), rank, i);
            rank++;
        }
        query = query.rename("x", "d1").expand("d0");
        FunctionEvaluator evaluator = modelsEvaluator.
                evaluatorOf("vespa_innerproduct_ranker");
        Tensor scores = evaluator.bind("query", query).
                bind("documents", batch.build()).evaluate();
        rank = 0;
        for (Hit h : result.hits()) {
            double score = scores.get(TensorAddress.of(rank));
            h.setRelevance(score);
            rank++;
        }
    }
}
