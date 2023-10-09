// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;


public class MyPostProcessingSearcher extends Searcher {

    private static final Tensor d0Tensor = Tensor.from("tensor<float>(d0[1]):[1]");

    private final ModelsEvaluator modelsEvaluator;

    public MyPostProcessingSearcher(ModelsEvaluator modelsEvaluator) {
        this.modelsEvaluator = modelsEvaluator;
    }

    @Override
    public Result search(Query query, Execution execution) {

        // Pass search through and be sure to fill results
        Result result = execution.search(query);
        execution.fill(result);

        // Create evaluator
        FunctionEvaluator evaluator = modelsEvaluator.evaluatorOf("pairwise_ranker");

        // Get query embedding
        Tensor queryEmbedding = query.getRanking().getFeatures().getTensor("query(embedding)").get().rename("x","d1");

        // Build model input by using the batch dimension - compare all documents against each other and the query
        Tensor queryBatch = null;
        Tensor doc1Batch = null;
        Tensor doc2Batch = null;
        int hitCount = result.getHitCount();
        for (int i1 = 0; i1 < hitCount; ++i1) {
            for (int i2 = 0; i2 < hitCount; ++i2) {
                if (i1 == i2) continue;  // don't compare against self
                Tensor d1 = ((Tensor) result.hits().get(i1).getField("embedding")).rename("x","d1");
                Tensor d2 = ((Tensor) result.hits().get(i2).getField("embedding")).rename("x","d1");
                queryBatch = concat(queryBatch, queryEmbedding, "d0");
                doc1Batch = concat(doc1Batch, d1, "d0");
                doc2Batch = concat(doc2Batch, d2, "d0");
            }
        }

        // Evaluate model once - for 10 hits output is size (d0[90],d1[1])
        Tensor output = evaluator.bind("query", queryBatch).bind("doc1", doc1Batch).bind("doc2", doc2Batch).evaluate();

        // Set final relevance score to the number of times document scores above some threshold,
        // e.g. probability > 0.5 that doc1 ranks above doc2.
        Tensor threshold = output.larger(output.avg());
        for (int hit = 0; hit < hitCount; ++hit) {
            Tensor mask = Util.evaluate("tensor(d0[" + threshold.size() + "])((d0 >= " + hit * (hitCount-1) + ") && (d0 < " + (hit+1)*(hitCount-1) + "))");
            result.hits().get(hit).setRelevance(mask.multiply(threshold).sum().asDouble());
        }

        // Perform the re-ordering
        result.hits().sort();
        return result;
    }

    private static Tensor concat(Tensor t1, Tensor t2, String dimension) {
        if (t1 == null) {
            return t2.multiply(d0Tensor);
        }
        return t1.concat(t2, dimension);
    }

}
