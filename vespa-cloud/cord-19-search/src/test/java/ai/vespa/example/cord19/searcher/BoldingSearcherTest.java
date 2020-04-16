package ai.vespa.example.cord19.searcher;

import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BoldingSearcherTest {

    @Test
    public void testThatStopWordIsAnnotedAsFilter(){
        Query input = new Query("?query=basic+reproduction+numbers+For+covid-19+IN+%22south+korea%22&type=any");
        Result result = execute(input, new BoldingSearcher());
        assertEquals("OR basic reproduction numbers |For (AND covid 19) |IN \"south korea\"",
                result.getQuery().getModel().getQueryTree().toString());

    }
    @Test
    public void testThatPhraseUntouched(){
        Query input = new Query("?query=temperature+%22impact+on%22+viral+transmission&type=any");
        Result result = execute(input, new BoldingSearcher());
        assertEquals("OR temperature \"impact on\" viral transmission",
                result.getQuery().getModel().getQueryTree().toString());

    }

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }
}
