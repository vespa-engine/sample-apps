// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.example;

import com.yahoo.prelude.query.AndItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.NullItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.query.QueryTree;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;

/**
 * An example searcher which modifies both the query and the returned result.
 */
public class ExampleSearcher extends Searcher {

    @Override
    public Result search(Query query, Execution execution) {
        // modify the query (add a term if it's empty)
        if (query.getModel().getQueryTree().getRoot() instanceof NullItem)
            query.getModel().getQueryTree().and(new WordItem("hello", "title"));

        // pass it down the chain to get a result
        Result result = execution.search(query);

        // process the result (add a synthetic hit)
        result.hits().add(new Hit("test:hit", 1.0));

        // return the result up the chain
        return result;
    }

}
