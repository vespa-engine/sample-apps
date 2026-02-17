// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.yql.MinimalQueryInserter;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static java.net.URLEncoder.encode;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SetFilterSearcherTest {

    @Test
    public void test_set_filter()   {
        Chain<Searcher> myChain = new Chain<>(new MinimalQueryInserter(), new SetFilterSearcher());
        Execution.Context context = Execution.Context.createContextStub();
        Execution execution = new Execution(myChain, context);
        String yql = encode("select * from sources * where artist contains \"metallica\"", StandardCharsets.UTF_8);
        Query query = new Query("/search/?yql=" + yql + "&set-filter=2018,2017&set-filter-field-name=year");
        Result result = execution.search(query);
        assertEquals("query \'AND artist:metallica |WEIGHTEDSET year{[1]:\"2018\",[1]:\"2017\"}\'",
                     result.getQuery().toString());
    }
}
