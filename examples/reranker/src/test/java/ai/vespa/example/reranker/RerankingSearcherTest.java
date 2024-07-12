// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.reranker;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author bratseth
 */
class RerankingSearcherTest {

    @Test
    void testRerankingSearcher() {
        var tester = new RerankingTester();

        Result result = tester.execution().search(new Query("?yql=select%20*%20from%20music%20where%20album%20contains%20%27to%27&passthrough=foo"));
        assertEquals(1.1, result.hits().get(0).getRelevance().getScore());
        assertEquals(1.1, result.hits().get(1).getRelevance().getScore());
        assertEquals(1.0, result.hits().get(2).getRelevance().getScore());
    }

}
