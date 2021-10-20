// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package application;

import com.google.gson.Gson;
import com.yahoo.vespa.http.client.FeedClient;
import com.yahoo.vespa.http.client.FeedClientFactory;
import com.yahoo.vespa.http.client.SimpleLoggerResultCallback;
import com.yahoo.vespa.http.client.config.Cluster;
import com.yahoo.vespa.http.client.config.ConnectionParams;
import com.yahoo.vespa.http.client.config.Endpoint;
import com.yahoo.vespa.http.client.config.FeedParams;
import com.yahoo.vespa.http.client.config.SessionParams;
import java.util.logging.Level;
import java.util.logging.Logger;
import json.Album;
import json.ImmutableTopLevelPut;
import json.TopLevelPut;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class VespaDataFeeder extends Thread {

    private final FeedClient feedClient;
    private final Gson gson = new Gson();
    private final BlockingQueue<Album> queue;
    private static final String ID_FORMAT = "id:mynamespace:music::%d";
    private final AtomicInteger pending = new AtomicInteger(0);
    private boolean shouldRun = true;
    private final Logger logger = Logger.getLogger(VespaDataFeeder.class.getName());


    VespaDataFeeder(BlockingQueue<Album> queue) {

        SessionParams sessionParams = new SessionParams.Builder()
                .addCluster(new Cluster.Builder().addEndpoint(Endpoint.create("vespa", 8080, false)).build())
                .setConnectionParams(new ConnectionParams.Builder().build())
                .setFeedParams(new FeedParams.Builder()
                        .setDataFormat(FeedParams.DataFormat.JSON_UTF8)
                        .build())
                .build();
        this.feedClient = FeedClientFactory.create(sessionParams, new SimpleLoggerResultCallback(this.pending, 100));

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
            feedClient.stream(id, gson.toJson(newAlbum));
        }
    }
}
