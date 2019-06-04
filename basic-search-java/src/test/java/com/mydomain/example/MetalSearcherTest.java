// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.example;

import com.yahoo.application.Application;
import com.yahoo.application.Networking;
import com.yahoo.application.container.Search;
import com.yahoo.component.ComponentSpecification;
import com.yahoo.component.chain.Chain;
import com.yahoo.config.model.ConfigModelRepo;
import com.yahoo.container.Container;
import com.yahoo.jdisc.handler.RequestHandler;
import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.handler.SearchHandler;
import com.yahoo.search.searchchain.Execution;

import com.yahoo.search.searchchain.SearchChain;
import com.yahoo.search.searchchain.SearchChainRegistry;
import com.yahoo.search.yql.MinimalQueryInserter;
import com.yahoo.yolean.chain.ChainBuilder;
import org.junit.Before;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.util.Iterator;
import java.util.List;

import static com.yahoo.component.ComponentSpecification.fromString;
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

    Query metalQuery;

    /**
     *
     */
    @Before
    public void initQuery() {
        metalQuery = new Query("/search/?yql=" +
                encode("select * from sources * where artist contains \"hetfield\" and title contains\"master of puppets\";",
                        StandardCharsets.UTF_8));
        metalQuery.setTraceLevel(6);
    }


    @Test
    public void testAddedOrTerm() {
        Chain<Searcher> myChain = new Chain<>(new MinimalQueryInserter(), new MetalSearcher());  // added to chain in this order
        Execution.Context context = Execution.Context.createContextStub();
        Execution execution = new Execution(myChain, context);

        Result result = execution.search(metalQuery);
        System.out.println(result.getContext(false).getTrace());

        assertAddedOrTerm(metalQuery.getModel().getQueryTree().getRoot());
    }

    @Test
    public void testAddedOrTerm2() {

        try (Application app = Application.fromApplicationPackage(
                FileSystems.getDefault().getPath("src/main/application"),
                Networking.disable)) {
            Search search = app.getJDisc("default").search();
            Result result = search.process(ComponentSpecification.fromString("mychain"), metalQuery);
            System.out.println(result.getContext(false).getTrace());

            assertAddedOrTerm(metalQuery.getModel().getQueryTree().getRoot());
        }
    }

    private void assertAddedOrTerm(Item root) {
        // Assert that an OR term is added to the root, with album:metal as one of the or-terms:
        assertTrue(root instanceof OrItem);
        for (Iterator<Item> iter = ((CompositeItem)root).getItemIterator(); iter.hasNext(); ) {
            Item item = iter.next();
            if (item instanceof WordItem) {
                assertEquals(item.toString(), "album:metal");
            }
        }
    }
}
