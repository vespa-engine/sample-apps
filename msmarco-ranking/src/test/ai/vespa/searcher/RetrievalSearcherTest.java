// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import com.yahoo.component.chain.Chain;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import org.junit.Test;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;

public class RetrievalSearcherTest {

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

    private void test_query_tree(String queryString, String expected,Searcher searcher){
        Query query = new Query("?query=" + URLEncoder.encode(queryString, StandardCharsets.UTF_8));
        Result r = execute(query,searcher);
        assertEquals("Not expected WAND query", expected,
                r.getQuery().getModel().getQueryTree().toString());
    }

    @Test
    public void test_queries() {
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics());
        test_query_tree("this is a test query",
                "WEAKAND(10) default:this default:is default:a default:test default:query",searcher);
        test_query_tree("with some +.% not ",
                "WEAKAND(10) default:with default:some default:not",searcher);
        test_query_tree("a number+:123  ?",
                "WEAKAND(10) default:a default:number default:123",searcher);
    }

    @Test
    public void test_params() {
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics());
        Query query = new Query("?query=a+test&wand.field=foo&wand.hits=100");
        Result result = execute(query,searcher);
        assertEquals("Not expected WAND query",
                "WEAKAND(100) foo:a foo:test", result.getQuery().getModel().getQueryTree().toString());

    }
}
