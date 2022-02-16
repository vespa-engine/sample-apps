// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher.colbert;

import ai.vespa.colbert.ColbertConfig;
import ai.vespa.models.evaluation.ModelsEvaluator;
import ai.vespa.searcher.RetrievalModelSearcher;
import ai.vespa.TokenizerFactory;
import com.yahoo.component.chain.Chain;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.Query;
import com.yahoo.tensor.Tensor;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class ColbertSearcherTest {

    private static final ColbertConfig colbertConfig;
    private static WordPieceEmbedder tokenizer;

    private static final int DIM = 32;
    private static final int MAX_QUERY_LENGHT = 32;

    static {
        tokenizer = TokenizerFactory.getEmbedder();
        ColbertConfig.Builder colbertBuilder = new ColbertConfig.Builder();
        colbertBuilder.dim(DIM).max_query_length(MAX_QUERY_LENGHT);
        colbertConfig = colbertBuilder.build();
    }

    @Test
    public void testColBERTSearcher() {
        RetrievalModelSearcher retrievalModelSearcher = new RetrievalModelSearcher(new SimpleLinguistics(),tokenizer);
        ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models/");
        ColBERTSearcher colBERTSearcher = new ColBERTSearcher(colbertConfig,evaluator);
        Query query = new Query("?query=what+was+the+impact+of+the+manhattan+project");
        Result result = execute(query, retrievalModelSearcher,colBERTSearcher);
        assertEquals(1, result.getConcreteHitCount());
        assertEquals(1, result.getTotalHitCount());
        Hit tensorHit = result.hits().get(0);
        assertEquals("colbert",tensorHit.getSource());
        Tensor colBertTensor = (Tensor)tensorHit.getField("tensor");
        assertNotNull(colBertTensor);
        assertEquals(32*32, colBertTensor.size()); //32 query terms x 32 dim
    }

    @Test
    public void testColBERTSearcherLong() {
        RetrievalModelSearcher retrievalModelSearcher = new RetrievalModelSearcher(new SimpleLinguistics(),tokenizer);
        ModelsEvaluator evaluator = ModelsEvaluatorTester.create("src/main/application/models/");
        ColBERTSearcher colBERTSearcher = new ColBERTSearcher(colbertConfig,evaluator);
        Query query = new Query("?query=________%20disparity%20refers%20to%20the%20slightly%20different%20view%20of%20the%20world%20that%20each%20eye%20receives.cyclopeanbinocularmonoculartrichromatic");
        Result result = execute(query, retrievalModelSearcher,colBERTSearcher);
        assertEquals(1, result.getConcreteHitCount());
        assertEquals(1, result.getTotalHitCount());
        Hit tensorHit = result.hits().get(0);
        assertEquals("colbert",tensorHit.getSource());
        Tensor colBertTensor = (Tensor)tensorHit.getField("tensor");
        assertNotNull(colBertTensor);
        assertEquals(32*32, colBertTensor.size()); //32 query terms x 32 dim
    }

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }
}


