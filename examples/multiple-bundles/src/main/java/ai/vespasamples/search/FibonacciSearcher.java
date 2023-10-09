// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespasamples.search;

import ai.vespasamples.lib.FibonacciProducer;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;

import java.util.logging.Logger;

/**
 * A searcher using an API from a library bundle in the same application.
 */
public class FibonacciSearcher extends Searcher {
    private static final Logger log = Logger.getLogger(FibonacciSearcher.class.getName());

    private final FibonacciProducer producer;

    private long numCalls = 0;

    public FibonacciSearcher(FibonacciProducer producer) {
        this.producer = producer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        Result result = execution.search(query); // Pass on to the next searcher to get results
        Hit hit = new Hit("test");
        hit.setField("message", "fib(" + numCalls++ + ") = " + producer.getNext());
        result.hits().add(hit);
        return result;
    }

    @Override
    public void deconstruct() {
        producer.stop();
    }

}
