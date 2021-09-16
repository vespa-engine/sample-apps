// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import ai.vespa.models.evaluation.ModelsEvaluator;
import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.component.chain.Chain;
import com.yahoo.data.access.slime.SlimeAdapter;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.slime.Cursor;
import com.yahoo.slime.Slime;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.serialization.TypedBinaryFormat;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;


import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assertions.assertNotNull;


public class QueryEmbeddingSearcherTest {



    @Test
    public void testEmbeddingSearcher() {
        Query query = new Query("?query=what+was+the+impact+of+the+manhattan+project");
        RetrievalModelSearcher retrievalModelSearcher = new RetrievalModelSearcher(new SimpleLinguistics(),TokenizerFactory.getTokenizer());
        ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models/");
        QueryEmbeddingSearcher searcher = new QueryEmbeddingSearcher(evaluator);
        Result result = execute(query, retrievalModelSearcher,searcher);
        assertEquals(1, result.getConcreteHitCount());
        assertEquals(1,result.getTotalHitCount());

        Hit tensorHit = result.hits().get(0);
        assertEquals("embedding", tensorHit.getSource());
        Tensor embeddingTensor = (Tensor)tensorHit.getField("tensor");
        System.out.println(embeddingTensor);
        assertNotNull(embeddingTensor);
        assertEquals(384, embeddingTensor.size());
    }

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }


}
