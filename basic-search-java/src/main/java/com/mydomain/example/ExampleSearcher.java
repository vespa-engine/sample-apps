// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
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
        // modify the query
        addItem(new WordItem("hello", "title"), query.getModel().getQueryTree());

        // pass it down the chain to get a resut
        Result result = execution.search(query);

        // process the result
        result.hits().add(new Hit("test:hit", 1.0));

        // return the result up the chain
        return result;
    }

    private void addItem(Item item, QueryTree queryTree) {
        if (queryTree.getRoot() instanceof NullItem) { // No existing query
            queryTree.setRoot(item);
        }
        else if (queryTree.getRoot() instanceof WordItem) { // Existing query is a single word
            AndItem and = new AndItem();
            and.addItem(queryTree.getRoot());
            and.addItem(item);
            queryTree.setRoot(and);
        }
        else if (queryTree.getRoot() instanceof AndItem) { // Existing query is already an and
            ((AndItem) queryTree.getRoot()).addItem(item);
        }
        // Other cases (such as NotItem not handled here)
    }

}
