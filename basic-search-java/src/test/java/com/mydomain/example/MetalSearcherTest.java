// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.example;

import com.yahoo.component.chain.Chain;
import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;

import com.yahoo.search.yql.MinimalQueryInserter;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.Iterator;

import static java.net.URLEncoder.encode;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Unit test - demonstrates:
 * <ol>
 *     <li>Build queries from YQL</li>
 *     <li>How to get the query tree, and evaluate it</li>
 *     <li>Use of tracing</li>
 * </ol>
 */
public class MetalSearcherTest {

    @Test
    public void testAddedOrTerm() {
        Chain<Searcher> myChain = new Chain<>(new MinimalQueryInserter(), new MetalSearcher());  // added to chain in this order
        Execution.Context context = Execution.Context.createContextStub();
        Execution execution = new Execution(myChain, context);

        Query myQuery = new Query("/search/?yql=" +
                encode("select * from sources * where artist contains \"hetfield\" and title contains\"master of puppets\";",
                StandardCharsets.UTF_8));

        myQuery.setTraceLevel(6);
        Result result = execution.search(myQuery);
        System.out.println(result.getContext(false).getTrace());

        // Assert that an OR term is added to the root, with album:metal as one of the or-terms:
        Item root = myQuery.getModel().getQueryTree().getRoot();
        assertTrue(root instanceof OrItem);
        for (Iterator<Item> iter = ((CompositeItem)root).getItemIterator(); iter.hasNext(); ) {
            Item item = iter.next();
            if (item instanceof WordItem) {
                assertEquals(item.toString(), "album:metal");
            }
        }
    }
}
