// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19.searcher;

import com.yahoo.component.chain.Chain;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author bratseth
 */
public class RelatedPaperSearcherANNTest {
    
    private final String expectedNNOR =
            "(OR NEAREST_NEIGHBOR {field=abstract_embedding,queryTensorName=abstract_vector,hnsw.exploreAdditionalHits=0,approximate=false,targetNumHits=100} " +
                    "NEAREST_NEIGHBOR {field=title_embedding,queryTensorName=title_vector,hnsw.exploreAdditionalHits=0,approximate=false,targetNumHits=100})";
    @Test
    public void testNoopIfNoRelated_to() {
        Query original = new Query("?query=foo%20bar").clone();
        Result result = execute(original, new RelatedPaperSearcherANN(), new MockBackend());
        assertEquals(original, result.getQuery());
    }

    @Test
    public void testRelated_toTermAddsNearestNeighborTermAndArticleFilter() {
        Query query = new Query("?query=foo%20bar%20related_to:123");
        Result result = execute(query, new RelatedPaperSearcherANN(), new MockBackend());
        assertEquals("+(AND (AND foo bar) " + expectedNNOR + ") -id:123",
                     result.getQuery().getModel().getQueryTree().toString());
    }

    @Test
    public void testRelated_toTermAddsNearestNeighborTermAndArticleFilterWithRankItem() {
        Query query = new Query("?query=covid-19+%2B%22south+korea%22+%2Brelated_to:123&type=any");
        Result result = execute(query, new RelatedPaperSearcherANN(), new MockBackend());
        assertEquals("+(AND (RANK (AND \"south korea\") (AND covid 19)) " + expectedNNOR + ") -id:123",
                     result.getQuery().getModel().getQueryTree().toString());
    }

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

    /** Handles queries fetching a specific article */
    private static class MockBackend extends Searcher {

        @Override
        public Result search(Query query, Execution execution) {
            if (isArticleRequest(query)) {
                Result result = execution.search(query);
                result.setTotalHitCount(1);
                Hit articleHit = new Hit("ignored", 1.0);
                articleHit.setField("title_embedding", mockEmbedding());
                articleHit.setField("abstract_embedding", mockEmbedding());
                result.hits().add(articleHit);
                return result;
            }
            else {
                return execution.search(query);
            }
        }

        private Tensor mockEmbedding() {
            Tensor.Builder b = Tensor.Builder.of(TensorType.fromSpec("tensor<float>(x[768])"));
            for (long i = 0; i < 768; i++)
                b.cell(Math.random(), i);
            return b.build();
        }

        private boolean isArticleRequest(Query query) {
            if (query.getHits() != 1) return false;
            if ( ! (query.getModel().getQueryTree().getRoot() instanceof WordItem)) return false;
            WordItem word = (WordItem)query.getModel().getQueryTree().getRoot();
            if ( ! "id".equals(word.getIndexName())) return false;
            return true;
        }

    }

}
