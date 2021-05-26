// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.component.chain.Chain;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.query.QueryTree;
import com.yahoo.search.searchchain.Execution;
import org.junit.Test;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.*;

public class RetrievalSearcherTest {

    private static BertTokenizer tokenizer;
    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(512);
        BertModelConfig bertModelConfig = builder.build();
        try {
            tokenizer = new BertTokenizer(bertModelConfig, new SimpleLinguistics());
        } catch (IOException e) {
            fail("IO Error during bert model read");
        }
    }

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
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics(),tokenizer);
        test_query_tree("this is a test query",
                "WEAKAND(10) default:this default:is default:a default:test default:query",searcher);
        test_query_tree("with some +.% not ",
                "WEAKAND(10) default:with default:some default:not",searcher);
        test_query_tree("a number+:123  ?",
                "WEAKAND(10) default:a default:number default:123",searcher);
    }

    @Test
    public void test_sparse_params() {
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics(),tokenizer);
        Query query = new Query("?query=a+test&wand.field=foo&wand.hits=100");
        Result result = execute(query,searcher);
        assertEquals("Not expected WAND query",
                "WEAKAND(100) foo:a foo:test", result.getQuery().getModel().getQueryTree().toString());
    }

    @Test
    public void test_dense_params() {
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics(), tokenizer);
        Query query = new Query("?query=a+test&retriever=dense&ann.hits=98");
        Result result = execute(query,searcher);
        QueryTree tree = result.getQuery().getModel().getQueryTree();
        Item item = tree.getRoot();
        if (!(item instanceof NearestNeighborItem))
            fail("Expected nearest neighbor item");
        NearestNeighborItem nn = (NearestNeighborItem)item;
        assertTrue(nn.getAllowApproximate());
        assertEquals(98, nn.getTargetNumHits());
        assertEquals("mini_document_embedding",nn.getIndexName());
        assertEquals("query_embedding",nn.getQueryTensorName());
    }


}
