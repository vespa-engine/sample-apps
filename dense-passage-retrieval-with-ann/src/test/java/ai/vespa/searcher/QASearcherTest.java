// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;



import com.yahoo.component.chain.Chain;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.Query;


public class QASearcherTest {

    private Result execute(Query query, Searcher... searcher) {
        Execution execution = new Execution(new Chain<>(searcher), Execution.Context.createContextStub());
        return execution.search(query);
    }

    private static class MockBackend extends Searcher {
        @Override
        public Result search(Query query, Execution execution) {
            return execution.search(query);
        }
    }
}


