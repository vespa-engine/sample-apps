// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19.searcher;

import com.yahoo.component.chain.Chain;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author bratseth
 */
public class RelatedArticlesByWeakAndSearcherTest {

    private final String titleWeakAndItem =
            "(WEAKAND(100) default:title!150)";

    private final String titleAndAbstractWeakAndTerm =
            "(WEAKAND(100) default:title!150 default:fine default:abstract)";

    @Test
    public void testNoopIfNoRelated_to() {
        Query original = new Query("?query=foo%20bar").clone();
        Result result = execute(original, new RelatedArticlesByWeakAndSearcher(new SimpleLinguistics()), new MockBackend());
        assertEquals(original, result.getQuery());
    }

    @Test
    public void testRelatedToTitleOnly() {
        Query query = new Query("?query=covid-19+%2B%22south+korea%22+%2Brelated_to:123&type=any&use-abstract=false");
        Result result = execute(query, new RelatedArticlesByWeakAndSearcher(new SimpleLinguistics()), new MockBackend());
        assertEquals("+(AND (RANK (AND \"south korea\") (AND covid 19)) " + titleWeakAndItem + ") -id:123",
                     result.getQuery().getModel().getQueryTree().toString());
    }

    @Test
    public void testRelatedToTitleAndAbstract() {
        Query query = new Query("?query=covid-19+%2B%22south+korea%22+%2Brelated_to:123&type=any&use-abstract=true");
        Result result = execute(query, new RelatedArticlesByWeakAndSearcher(new SimpleLinguistics()), new MockBackend());
        assertEquals("+(AND (RANK (AND \"south korea\") (AND covid 19)) " + titleAndAbstractWeakAndTerm + ") -id:123",
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
                articleHit.setField("title-full", "my title");
                articleHit.setField("abstract-full", "my fine abstract");
                result.hits().add(articleHit);
                return result;
            }
            else {
                return execution.search(query);
            }
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
