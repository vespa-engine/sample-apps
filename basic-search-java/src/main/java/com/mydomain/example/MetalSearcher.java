// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.example;

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

import java.util.Arrays;
import java.util.List;

/**
 * Searcher which adds an OR-term to queries with Metal intent - demonstrates:
 * <ol>
 *     <li>How to get the query tree, and modify it</li>
 *     <li>Use of tracing</li>
 * </ol>
 */
public class MetalSearcher extends Searcher {

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
        List<String> names = Arrays.asList("hetfield", "metallica", "pantera");
        for (IndexedItem posItem : QueryTree.getPositiveTerms(items) ) {
            return names.contains(posItem.getIndexedString());
        }
        return false;
    }
}
