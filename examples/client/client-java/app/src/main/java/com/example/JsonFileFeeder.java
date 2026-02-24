// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.example;

import ai.vespa.feed.client.DocumentId;
import ai.vespa.feed.client.FeedClient;
import ai.vespa.feed.client.FeedClientBuilder;
import ai.vespa.feed.client.FeedException;
import ai.vespa.feed.client.JsonFeeder;
import ai.vespa.feed.client.Result;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

import javax.net.ssl.SSLContext;

/**
 * Sample feeder demonstrating how to programmatically feed to a Vespa cluster.
 */
class JsonFileFeeder implements Closeable {

    private final static Logger log = Logger.getLogger(JsonFileFeeder.class.getName());

    private final JsonFeeder jsonFeeder;

    static class ResultCallBack implements JsonFeeder.ResultCallback {

        final AtomicInteger resultsReceived = new AtomicInteger(0);
        final AtomicInteger errorsReceived = new AtomicInteger(0);
        final long startTimeMillis = System.currentTimeMillis();;

        @Override
        public void onNextResult(Result result, FeedException error) {
            resultsReceived.incrementAndGet();
            if (error != null) {
                log.warning("Problems with feeding document "
                        + error.documentId().map(DocumentId::toString).orElse("<unknown>")
                        + ": " + error);
                errorsReceived.incrementAndGet();
            }
        }

        @Override
        public void onError(FeedException error) {
            log.severe("Feeding failed fatally: " + error.getMessage());
        }

        @Override
        public void onComplete() {
            log.info("Feeding completed");
        }

        void dumpStatsToLog() {
            long timeSpentMillis = System.currentTimeMillis() - startTimeMillis;
            log.info("Received in total " + resultsReceived.get() + ", " + errorsReceived.get() + " errors.");
            log.info("Time spent receiving is " + timeSpentMillis + " ms.");
            double okRate = (double)resultsReceived.get() * 1000.0 / (double)timeSpentMillis;
            log.info("OK Rate: " + okRate + "/s");
        }

    }

    JsonFileFeeder(FeedClient feedClient) {
        this.jsonFeeder = JsonFeeder.builder(feedClient)
                .withTimeout(Duration.ofSeconds(30))
                .build();
    }

    /**
     * Feed all operations from a stream.
     *
     * @param stream The input stream to read operations from (JSON array containing one or more document operations).
     */
    void batchFeed(InputStream stream) {
        ResultCallBack callback = new ResultCallBack();
        log.info("Starting feed");
        CompletableFuture<Void> promise = jsonFeeder.feedMany(stream, callback);
        promise.join(); // wait for feeding to complete
        callback.dumpStatsToLog();
    }

    @Override
    public void close() throws IOException {
        jsonFeeder.close();
    }
}
