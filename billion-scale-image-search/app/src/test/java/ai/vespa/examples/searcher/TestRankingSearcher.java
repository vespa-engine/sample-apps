// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.searcher;

import ai.vespa.examples.BPETokenizer;
import ai.vespa.examples.Utils;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.Chain;
import com.yahoo.config.FileReference;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import java.util.Optional;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestRankingSearcher {

    @Test
    public void test_searcher_chain() {
        FileReference file = new FileReference("src/main/application/files/bpe_simple_vocab_16e6.txt.gz");
        ai.vespa.examples.BpeTokenizerConfig config =
                new ai.vespa.examples.BpeTokenizerConfig.Builder().contextlength(77).vocabulary(file).build();
        BPETokenizer tokenizer = new BPETokenizer(config);
        ModelsEvaluator modelsEvaluator = ModelsEvaluatorTester.create("src/main/application/models");

        CLIPEmbeddingSearcher clipSearcher = new CLIPEmbeddingSearcher(modelsEvaluator, tokenizer);
        RankingSearcher rankingSearcher = new RankingSearcher(modelsEvaluator);
        MockBackEndSearcher backEndSearcher = new MockBackEndSearcher();

        Query query = new Query("/search/?prompt=Fairy+in+Forest");
        Result r = execute(query, clipSearcher, rankingSearcher, backEndSearcher);
        Optional<Tensor> t = query.getRanking().getFeatures().getTensor("query(q)");
        assertTrue(t.isPresent());
        assertEquals("id:laion:image::-5843246909686162287", r.hits().get(0).getId().toString());
        assertEquals(24,r.getConcreteHitCount());
        assertTrue(r.hits().get(0).getRelevance().getScore() > 0.2 );
    }

    private static class MockBackEndSearcher extends Searcher {
        @Override
        public Result search(Query query, Execution execution) {
            try {
                return Utils.readTestResponse(query);
            } catch (Exception e) {
                throw new RuntimeException("Failed to read test resoruces");
            }
        }
    }


    private static Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }
}
