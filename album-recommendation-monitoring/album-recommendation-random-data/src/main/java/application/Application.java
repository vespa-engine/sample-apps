// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package application;

import java.io.IOException;
import java.net.MalformedURLException;
import java.util.logging.Level;
import java.util.logging.Logger;
import json.Album;

import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class Application {
    private final Logger logger = Logger.getLogger(Application.class.getName());
    private static final int RUNS_PER_SECOND = 100;
    private final Random random = new Random();
    private final AtomicInteger pendingQueryRequests;
    RandomAlbumGenerator albumGenerator;
    BlockingQueue<Album> queue;
    VespaDataFeeder dataFeeder;
    VespaQueryFeeder queryFeeder;
    private double pushProbability = 0.05;
    private double queryProbability = 0.05;
    private boolean isGrowing = true;

    Application() {
        albumGenerator = new RandomAlbumGenerator();
        queue = new LinkedBlockingQueue<>();
        pendingQueryRequests = new AtomicInteger(0);
    }

    private void createFeeders() {
        dataFeeder = new VespaDataFeeder(queue);
        queryFeeder = new VespaQueryFeeder(pendingQueryRequests);
    }

    private boolean isConnection200(URL url) {
        try {
            return ((HttpURLConnection) url.openConnection()).getResponseCode() == 200;
        } catch (IOException e) {
            return false;
        }

    }

    private void waitForVespa() {
        int attempts = 0;
        URL vespa = null;
        try {
            vespa = new URL("http://vespa:8080/ApplicationStatus");
        } catch (MalformedURLException e) {
            logger.log(Level.SEVERE, e.getMessage());
            System.exit(1);
        }
        boolean success = isConnection200(vespa);
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
            success = isConnection200(vespa);
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
                (long) 30 * 1000,
                (long) (1000.0 / (double) RUNS_PER_SECOND)
        );

        new Timer().scheduleAtFixedRate(new TimerTask() {
                @Override
                public void run() {
                    if (shouldQuery()) pendingQueryRequests.incrementAndGet();
                }},
                (long) 60 * 1000,
                (long) (1000.0 / (double) RUNS_PER_SECOND)
        );

        new Timer().scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                updatePushProbability();
                updateQueryProbability();
            }},
                (long) 10*1000,
                (long) 10*1000
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
