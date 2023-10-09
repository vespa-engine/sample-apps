// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.application;

import ai.vespa.examples.json.Album;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Application {
    private static final Logger logger = Logger.getLogger(Application.class.getName());
    private static final int RUNS_PER_SECOND = 100;
    static String ENDPOINT = "http://vespa-container:8080";

    private final Random random = new Random();
    private final AtomicInteger pendingQueryRequests = new AtomicInteger(0);
    private final RandomAlbumGenerator albumGenerator = new RandomAlbumGenerator();
    private final BlockingQueue<Album> queue = new LinkedBlockingQueue<>();

    private VespaDataFeeder dataFeeder;
    private VespaQueryFeeder queryFeeder;
    private double pushProbability = 0.05;
    private double queryProbability = 0.05;
    private boolean isGrowing = true;

    private void createFeeders() {
        dataFeeder = new VespaDataFeeder(queue);
        queryFeeder = new VespaQueryFeeder(pendingQueryRequests);
    }

    private boolean isConnection200(HttpClient client) {
        try {
            HttpRequest request = HttpRequest.newBuilder().uri(URI.create(ENDPOINT + "/ApplicationStatus")).build();
            return client.send(request, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    private void waitForVespa() {
        int attempts = 0;
        HttpClient client = HttpClient.newBuilder().build();

        boolean success = isConnection200(client);
        while (!success) {
            logger.info("Unable to connect to Vespa, trying again in 20 seconds");
            attempts++;
            if (attempts >= 15) {
                logger.log(Level.SEVERE, "Failure. Cannot establish connection to Vespa");
                System.exit(1);
            }
            try {
                Thread.sleep(20000);
            } catch (InterruptedException e) {
                logger.log(Level.SEVERE, e.getMessage());
                Thread.currentThread().interrupt();
            }
            success = isConnection200(client);
        }
    }

    public void start() {
        waitForVespa();
        createFeeders();

        dataFeeder.start();
        queryFeeder.start();

        new Timer().scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if (shouldPush()) queue.add(albumGenerator.getRandomAlbum());
            }},
                30_000,
                (long) (1000d / RUNS_PER_SECOND)
        );

        new Timer().scheduleAtFixedRate(new TimerTask() {
                @Override
                public void run() {
                    if (shouldQuery()) pendingQueryRequests.incrementAndGet();
                }},
                60_000,
                (long) (1000d / RUNS_PER_SECOND)
        );

        new Timer().scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                updatePushProbability();
                updateQueryProbability();
            }},
                10_000,
                10_000
        );
    }

    private void updatePushProbability() {
        pushProbability += isGrowing ? random.nextDouble() * 0.1 : -random.nextDouble() * 0.1;
        if (pushProbability >= 0.95) {
            pushProbability = 0.95;
            isGrowing = false;
        } else if (pushProbability <= 0.05) {
            pushProbability = 0.05;
            isGrowing = true;
        }
    }

    private void updateQueryProbability() {
        queryProbability = (random.nextDouble() * 0.4) + 0.4;
    }

    private boolean shouldPush() {
        return random.nextDouble() < pushProbability;
    }

    private boolean shouldQuery() {
        return random.nextDouble() < queryProbability;
    }


    public static void main(String[] args) {
        new Application().start();
    }
}
