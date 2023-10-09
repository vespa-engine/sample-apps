// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.application;

import ai.vespa.examples.json.ImmutableQuery;
import com.google.gson.Gson;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import static ai.vespa.examples.application.Application.ENDPOINT;

public class VespaQueryFeeder extends Thread {

    private static final Logger logger = Logger.getLogger(VespaQueryFeeder.class.getName());

    private final AtomicInteger pendingQueryRequests;
    private final HttpClient client;
    private final HttpRequest request;
    private volatile boolean shouldRun = true;

    VespaQueryFeeder(AtomicInteger pendingQueryRequests) {
        this.pendingQueryRequests = pendingQueryRequests;
        this.client = HttpClient.newBuilder().build();

        ImmutableQuery query = ImmutableQuery.builder().yql("SELECT * FROM SOURCES * WHERE year > 2000").build();
        this.request = HttpRequest.newBuilder()
                .uri(URI.create(ENDPOINT + "/search/"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(new Gson().toJson(query)))
                .build();
    }

    public void queryVespa() throws InterruptedException {
        try {
            String result = client.send(request, HttpResponse.BodyHandlers.ofString())
                    .body();
            logger.log(Level.FINE, result);
        } catch (IOException e) {
            logger.log(Level.SEVERE, e.getMessage());
        }
    }

    public void shutDown() {
        shouldRun = false;
    }

    @Override
    public void run() {
        try {
            while (shouldRun) {
                if (pendingQueryRequests.get() > 0) {
                    queryVespa();
                    pendingQueryRequests.decrementAndGet();
                }
            }
        } catch (InterruptedException e) {
            logger.log(Level.SEVERE, e.getMessage());
            Thread.currentThread().interrupt();
        }
    }
}
