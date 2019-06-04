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
import com.yahoo.search.query.Model;
import com.yahoo.search.query.QueryTree;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.yolean.chain.After;

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

    @Override
    public Result search(Query query, Execution execution) {
        Model model = query.getModel();
        QueryTree tree = model.getQueryTree();
        if (!tree.isEmpty()) {
            Item rootItem = tree.getRoot();
            if (isMetalQuery(rootItem)) {
                OrItem orItem = new OrItem();
                orItem.addItem(rootItem);
                orItem.addItem(new WordItem("metal", "album"));
                tree.setRoot(orItem);
            }
        }
        query.trace("Metal added", true, 2);
        return execution.search(query);
    }

    private boolean isMetalQuery(Item items) {
        for (IndexedItem posItem : QueryTree.getPositiveTerms(items) ) {
            return metalWords.contains(posItem.getIndexedString());
        }
        return false;
    }
}
