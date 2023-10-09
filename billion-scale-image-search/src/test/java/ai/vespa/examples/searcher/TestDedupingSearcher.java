// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.searcher;

import ai.vespa.examples.Utils;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestDedupingSearcher {

    @Test
    public void test_deduping() {
        DeDupingSearcherTester tester = new DeDupingSearcherTester();
        Query query = new Query("?query=foo");
        query.setHits(24);
        Result result = tester.execute(query);
        assertEquals(22,result.getConcreteHitCount());

        query.properties().set("collapse.enable", "false");
        result = tester.execute(query);
        assertEquals(24,result.getConcreteHitCount());


        query.properties().set("collapse.enable", "true");
        query.properties().set("collapse.similarity.threshold", 0.1);
        result = tester.execute(query);
        assertEquals(1,result.getConcreteHitCount());

        query.setHits(10);
        query.properties().set("collapse.enable", "true");
        query.properties().set("collapse.similarity.threshold", 0.9);
        result = tester.execute(query);
        assertEquals(10,result.getConcreteHitCount());
    }

    private static class DeDupingSearcherTester {
        DeDupingSearcher searcher;
        Chain<Searcher> chain;

        private DeDupingSearcherTester(){
            ModelsEvaluator eval = ModelsEvaluatorTester.create("src/main/application/models/");
            searcher = new DeDupingSearcher(eval);
            chain = new Chain<>(searcher, new MockBackend());
        }
        private Result execute(Query query) {
            Execution execution = new Execution(chain, Execution.Context.createContextStub());
            return execution.search(query);
        }
    }


    private static class MockBackend extends Searcher {
        @Override
        public Result search(Query query, Execution execution) {
            try {
                return Utils.readTestResponse(query);
            } catch(Exception e) {
                throw new RuntimeException("Failed to read test resoruces");
            }
        }
    }


}
