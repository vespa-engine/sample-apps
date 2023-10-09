// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.application;

import ai.vespa.examples.json.Album;
import ai.vespa.examples.json.ImmutableTopLevelPut;
import ai.vespa.examples.json.TopLevelPut;
import ai.vespa.feed.client.DocumentId;
import ai.vespa.feed.client.FeedClient;
import ai.vespa.feed.client.FeedClientBuilder;
import ai.vespa.feed.client.FeedException;
import ai.vespa.feed.client.OperationParameters;
import ai.vespa.feed.client.Result;
import com.google.gson.Gson;

import java.net.URI;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import static ai.vespa.examples.application.Application.ENDPOINT;

public class VespaDataFeeder extends Thread {

    private static final Logger logger = Logger.getLogger(VespaDataFeeder.class.getName());
    private static final String ID_FORMAT = "id:mynamespace:music::%d";

    private final Gson gson = new Gson();
    private final AtomicInteger pending = new AtomicInteger(0);
    private final FeedClient feedClient;
    private final BlockingQueue<Album> queue;
    private volatile boolean shouldRun = true;

    VespaDataFeeder(BlockingQueue<Album> queue) {
        this.feedClient = FeedClientBuilder.create(URI.create(ENDPOINT)).build();
        this.queue = queue;
    }

    public void shutDown() {
        this.shouldRun = false;
    }

    @Override
    public void run() {
        while (this.shouldRun || !queue.isEmpty()) {
            Album album = null;
            try {
                album = queue.take();
            } catch (InterruptedException e) {
                logger.log(Level.SEVERE, "Encountered exception while attempting to retrieve album");
                logger.log(Level.SEVERE, e.getMessage());
                Thread.currentThread().interrupt();
            }
            String id = String.format(ID_FORMAT, pending.incrementAndGet());
            assert album != null;
            TopLevelPut newAlbum = ImmutableTopLevelPut.builder()
                    .fields(album)
                    .put(id)
                    .build();
            pending.incrementAndGet();
            CompletableFuture<Result> promise = feedClient.put(DocumentId.of(id), gson.toJson(newAlbum), OperationParameters.empty());
            promise.whenComplete(this::onOperationComplete);
        }
    }

    private void onOperationComplete(Result result, Throwable error) {
        pending.decrementAndGet();
        if (error != null) {
            FeedException feedException = (FeedException) error;
            String docId = feedException.documentId().map(DocumentId::toString).orElse("<unknown-doc-id>");
            logger.log(Level.SEVERE, "Failed to feed " + docId, feedException);
        } else {
            logger.log(Level.FINE, "Successfully fed " + result.documentId());
        }
    }
}
