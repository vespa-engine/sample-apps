// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.example;

import com.google.inject.Inject;
import com.yahoo.prelude.query.IndexedItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.query.QueryTree;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.yolean.chain.After;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Searcher which adds an OR-term to queries with Metal intent - demonstrates:
 * <ol>
 *     <li>How to get the query tree, and modify it</li>
 *     <li>Use of tracing</li>
 * </ol>
 */
@After("MinimalQueryInserter")
public class MetalSearcher extends Searcher {

    private final List<String> metalWords;

    @Inject
    public MetalSearcher(MetalNamesConfig config) {
        metalWords = config.metalWords();
    }

    public MetalSearcher() {
        metalWords = new ArrayList<>();
    }

    @Override
    public Result search(Query query, Execution execution) {
        QueryTree tree = query.getModel().getQueryTree();
        if (isMetalQuery(tree)) {
            OrItem orItem = new OrItem();
            orItem.addItem(tree.getRoot());
            orItem.addItem(new WordItem("metal", "album"));
            tree.setRoot(orItem);
            query.trace("Metal added", true, 2);
        }

        Result result = execution.search(query);

        // result.hits().add(new Hit("test:hit", 1.0)); ToDo: expand example with a Searcher that adds hits

        return result;
    }

    private boolean isMetalQuery(Item items) {
        for (IndexedItem posItem : QueryTree.getPositiveTerms(items) ) {
            return metalWords.contains(posItem.getIndexedString());
        }
        return false;
    }

    private boolean isMetalQuery(QueryTree tree) {
        if (tree.isEmpty()) {
            return false;
        }
        for (IndexedItem posItem : QueryTree.getPositiveTerms(tree.getRoot()) ) {
            if (metalWords.contains(posItem.getIndexedString())) {
                return true;
            }
        }
        return false;
    }
}
