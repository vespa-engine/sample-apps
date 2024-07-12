// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.reranker;

import com.yahoo.component.chain.Chain;
import com.yahoo.container.jdisc.HttpRequest;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author bratseth
 */
class VespaSearcherTest {

    @Test
    void testVespaSearcher() {
        var tester = new RerankingTester();
        Result result = tester.executionOf(tester.vespaSearcher).search(new Query("?yql=select%20*%20from%20music%20where%20album%20contains%20%27to%27&passthrough=foo"));
        assertEquals(Map.of("ranking", "default"), tester.client.lastOverridingProperties);
        assertEquals("select * from music where album contains 'to'", tester.client.lastRequest.getProperty("yql"));
        assertEquals("foo", tester.client.lastRequest.getProperty("passthrough"));
        assertEquals(3, result.getConcreteHitCount());
        assertEquals(3, result.hits().size());
    }

}
