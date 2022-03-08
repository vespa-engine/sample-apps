// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

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
 * A searcher which adds an OR-term to queries with Metal intent.
 *
 * See https://docs.vespa.ai/en/searcher-development.html
 */
@After("MinimalQueryInserter")
public class MetalSearcher extends Searcher {

    private final List<String> metalWords;

    /**
     * Annotating the constructor with @Inject tells the container to use this constructor
     * when building the searcher.
     * 
     * <pre>MetalNamesConfig</pre> is automatically generated based on the
     * <pre>metal-names.def</pre> file found in resources/configdefinitions.
     *
     * See
     *  - https://docs.vespa.ai/en/jdisc/injecting-components.html
     *  - https://docs.vespa.ai/en/reference/config-files.html#config-definition-files
     * 
     * @param config the configuration object injected by the container.
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
     * Search method takes the query and an execution context. This can
     * manipulate the Query object, the Result coming back, issue multiple queries
     * in parallel or serially etc.
     */
    @Override
    public Result search(Query query, Execution execution) {
        if ( ! isMetalQuery(query)) return execution.search(query);

        QueryTree tree = query.getModel().getQueryTree();
        OrItem orItem = new OrItem();
        orItem.addItem(tree.getRoot());
        orItem.addItem(new WordItem("metal", "album"));
        tree.setRoot(orItem);
        query.trace("Metal added", true, 2);

        return execution.search(query);
    }

    private boolean isMetalQuery(Query query) {
        for (IndexedItem posItem : QueryTree.getPositiveTerms(query.getModel().getQueryTree().getRoot())) {
            if (metalWords.contains(posItem.getIndexedString()))
                return true;
        }
        return false;
    }

}
