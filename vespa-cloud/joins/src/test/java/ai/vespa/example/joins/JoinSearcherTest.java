package ai.vespa.example.joins;

import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import org.junit.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

public class JoinSearcherTest {

    @Test
    public void testJoinSearcher() {
        Query query = new Query();
        query.properties().set("a.type", "title");
        query.properties().set("a.start", 500L);
        query.properties().set("a.end", 1000L);
        query.properties().set("a.field", "text");
        query.properties().set("a.value", "text1");
        query.properties().set("b.type", "tag");
        query.properties().set("b.field", "tagId");
        query.properties().set("b.value", "1");
        MockBackend backend = new MockBackend();
        Result result = execute(query, new JoinSearcher(), backend);
        assertEquals(3, result.hits().size());
        assertHit("id1", "text1", "1", 200, 600, 300, 400, result.hits().get(0));
        assertHit("id2", "text1", "1", 700, 800, 700, 900, result.hits().get(1));
        assertHit("id2", "text1", "1", 700, 800, 950, 970, result.hits().get(2));
    }

    private void assertHit(String id, String text, String tagId, long start, long end, long tagStart, long tagEnd, Hit hit) {
        assertEquals(id,       hit.getField("id"));
        assertEquals(text,     hit.getField("text"));
        assertEquals(tagId,    hit.getField("tagId"));
        assertEquals(start,    hit.getField("start"));
        assertEquals(end,      hit.getField("end"));
        assertEquals(tagStart, hit.getField("tagStart"));
        assertEquals(tagEnd,   hit.getField("tagEnd"));
    }

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

    static class MockBackend extends Searcher {

        boolean gotAQuery = false;
        int bQueryCount = 0;

        @Override
        public Result search(Query query, Execution execution) {
            if (! gotAQuery) {
                gotAQuery = true;
                return handleAQuery(query);
            }
            else {
                return handleBQuery(query);
            }
        }

        private Result handleAQuery(Query query) {
            assertEquals(Set.of("title"), query.getModel().getSources());
            assertEquals("+(AND start:[;1000] text:text1) -end:[;500]", query.getModel().getQueryTree().getRoot().toString());
            Result result = new Result(query);
            result.hits().add(createAHit("title1", "id1", "text1", 200,  600));
            result.hits().add(createAHit("title2", "id2", "text1", 700,  800));
            return result;
        }

        private Result handleBQuery(Query query) {
            bQueryCount++;
            assertEquals(Set.of("tag"), query.getModel().getSources());

            if (bQueryCount == 1) {
                assertEquals("+(AND start:[;600] (AND tagId:1 id:id1)) -end:[;200]", query.getModel().getQueryTree().getRoot().toString());
                Result result = new Result(query);
                result.hits().add(createBHit("tag1", "id1", "1", 300, 400));
                return result;
            }
            else if (bQueryCount == 2) {
                assertEquals("+(AND start:[;800] (AND tagId:1 id:id2)) -end:[;700]", query.getModel().getQueryTree().getRoot().toString());
                Result result = new Result(query);
                result.hits().add(createBHit("tag2", "id2", "1", 700, 900));
                result.hits().add(createBHit("tag3", "id2", "1", 950, 970));
                return result;
            }
            else {
                fail("Got an unexpected third 'b' query: " + query);
                return null;
            }
        }

        private Hit createAHit(String hitId, String id, String text, long start, long end) {
            Hit hit = new Hit(hitId);
            hit.setField("id", id);
            hit.setField("text", text);
            hit.setField("start", start);
            hit.setField("end", end);
            return hit;
        }

        private Hit createBHit(String hitId, String id, String tagId, long start, long end) {
            Hit hit = new Hit(hitId);
            hit.setField("id", id);
            hit.setField("tagId", tagId);
            hit.setField("start", start);
            hit.setField("end", end);
            return hit;
        }

    }

}
