// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import ai.vespa.TokenizerFactory;
import ai.vespa.embedding.EmbeddingConfig;
import ai.vespa.embedding.DenseEmbedder;
import ai.vespa.models.evaluation.ModelsEvaluator;

import com.yahoo.component.chain.Chain;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;


public class QueryEmbeddingSearcherTest {
    static final DenseEmbedder embedder;

    static  {
        ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models/");
        EmbeddingConfig.Builder builder = new EmbeddingConfig.Builder();
        builder.max_document_length(256).max_query_length(32);
        builder.model_name("dense_encoder");
        embedder = new DenseEmbedder(evaluator,TokenizerFactory.getEmbedder(),builder.build());
    }

    @Test
    public void testEmbeddingSearcher() {
        Query query = new Query("?query=what+was+the+impact+of+the+manhattan+project");
        RetrievalModelSearcher retrievalModelSearcher =
                new RetrievalModelSearcher(new SimpleLinguistics(), TokenizerFactory.getEmbedder());
        QueryEmbeddingSearcher searcher = new QueryEmbeddingSearcher(embedder);
        Result result = execute(query, retrievalModelSearcher,searcher);
        assertEquals(1, result.getConcreteHitCount());
        assertEquals(1,result.getTotalHitCount());

        Hit tensorHit = result.hits().get(0);
        assertEquals("embedding", tensorHit.getSource());
        Tensor embeddingTensor = (Tensor)tensorHit.getField("tensor");
        assertNotNull(embeddingTensor);
        assertEquals(384, embeddingTensor.size());
    }

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }
}
