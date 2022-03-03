// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package application;

import com.google.gson.Gson;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import json.ImmutableQuery;

public class VespaQueryFeeder extends Thread {

    private final AtomicInteger pendingQueryRequests;
    Logger logger = Logger.getLogger(VespaQueryFeeder.class.getName());
    HttpClient client;
    HttpRequest request;
    private volatile boolean shouldRun = true;

    VespaQueryFeeder(AtomicInteger pendingQueryRequests) {
        this.pendingQueryRequests = pendingQueryRequests;

        client = HttpClient.newHttpClient();
        ImmutableQuery query = ImmutableQuery.builder().yql("SELECT * FROM SOURCES * WHERE year > 2000").build();

         request = HttpRequest.newBuilder()
                .uri(URI.create("http://vespa:8080/search/"))
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
