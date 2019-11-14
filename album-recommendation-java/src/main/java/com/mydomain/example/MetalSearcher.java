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

    /**
     * Annotating the constructor with @Inject tells the container which constructor to
     * use when building the searcher.
     * 
     * <pre>MetalNamesConfig</pre> is automatically generated based on the
     * <pre>metal-names.def</pre> file found in resources/configdefinitions.
     * 
     * @see https://docs.vespa.ai/documentation/jdisc/injecting-components.html
     * @see https://docs.vespa.ai/documentation/reference/config-files.html#config-definition-files
     * 
     * @param config The configuration object injected by the container.
     */
    @Inject
    public MetalSearcher(MetalNamesConfig config) {
        metalWords = config.metalWords();
    }

    /**
     * Default constructor typically used for tests, etc.
     */
    public MetalSearcher() {
        metalWords = new ArrayList<>();
    }

    /**
     * Search method takes the query and an execution context.  This method can
     * manipulate both the Query object and the Result object before passing it
     * further in the chain.
     * 
     * @see https://docs.vespa.ai/documentation/searcher-development.html
     */
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

        return execution.search(query);
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
