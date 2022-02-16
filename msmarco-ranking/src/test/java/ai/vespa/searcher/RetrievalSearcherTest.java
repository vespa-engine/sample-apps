// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import ai.vespa.TokenizerFactory;
import com.yahoo.component.chain.Chain;
import com.yahoo.language.detect.Detection;
import com.yahoo.language.detect.Detector;
import com.yahoo.language.detect.Hint;
import com.yahoo.language.opennlp.OpenNlpLinguistics;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.query.QueryTree;
import com.yahoo.search.searchchain.Execution;
import org.junit.jupiter.api.Test;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class RetrievalSearcherTest {
    static WordPieceEmbedder tokenizer;

    static {
        tokenizer = TokenizerFactory.getEmbedder();
    }

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

    private void test_query_tree(String queryString, String expected, Searcher searcher){
        Query query = new Query("?query=" + URLEncoder.encode(queryString, StandardCharsets.UTF_8));
        Result r = execute(query,searcher);
        assertEquals(expected, r.getQuery().getModel().getQueryTree().toString(), "Not expected WAND query");
    }

    @Test
    public void test_queries() {
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics(), tokenizer);
        test_query_tree("this is a test query",
                        "WEAKAND(10) default:this default:is default:a default:test default:query", searcher);
        test_query_tree("with some +.% not ",
                        "WEAKAND(10) default:with default:some default:not", searcher);
        test_query_tree("a number+:123  ?",
                        "WEAKAND(10) default:a default:number default:123", searcher);

    }

    @Test
    public void test_sparse_params() {
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics(), tokenizer);
        Query query = new Query("?query=a+test&wand.field=foo&wand.hits=100");
        Result result = execute(query, searcher);
        assertEquals("WEAKAND(100) foo:a foo:test", result.getQuery().getModel().getQueryTree().toString());
    }

    @Test
    public void test_language() {
        OpenNlpLinguistics linguistics = new OpenNlpLinguistics();
        Detector detect = linguistics.getDetector();
        String queryString = "Kosten für endlose Pools/Schwimmbad";
        Detection detection = detect.detect(queryString, Hint.newCountryHint("en"));
        assertEquals("de", detection.getLanguage().languageCode());
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics(), tokenizer);
        Query query = new Query("?query=" + URLEncoder.encode(queryString, StandardCharsets.UTF_8)
                + "&wand.field=foo&wand.hits=10&language=" + detection.getLanguage().languageCode());
        Result result = execute(query, searcher);
        assertEquals("WEAKAND(10) foo:kosten foo:fur foo:endlose foo:pools foo:schwimmbad", result.getQuery().getModel().getQueryTree().toString());
    }

    @Test
    public void test_dense_params() {
        RetrievalModelSearcher searcher = new RetrievalModelSearcher(new SimpleLinguistics(), tokenizer);
        Query query = new Query("?query=a+test&retriever=dense&ann.hits=98");
        Result result = execute(query, searcher);
        QueryTree tree = result.getQuery().getModel().getQueryTree();
        Item item = tree.getRoot();
        if (!(item instanceof NearestNeighborItem))
            fail("Expected nearest neighbor item");
        NearestNeighborItem nn = (NearestNeighborItem)item;
        assertTrue(nn.getAllowApproximate());
        assertEquals(98, nn.getTargetNumHits());
        assertEquals("mini_document_embedding", nn.getIndexName());
        assertEquals("query_embedding", nn.getQueryTensorName());
    }
}
