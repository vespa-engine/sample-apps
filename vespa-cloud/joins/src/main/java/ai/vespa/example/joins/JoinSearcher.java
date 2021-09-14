// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.EquivItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.PhraseItem;
import com.yahoo.prelude.query.TermItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.query.QueryTree;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.yolean.chain.After;

import java.util.List;
import java.util.Map;

@After("MinimalQueryInserter")
public class JoinSearcher extends Searcher {

    @Override
    public Result search(Query query, Execution execution) {
        return execution.search(query);
    }

}
