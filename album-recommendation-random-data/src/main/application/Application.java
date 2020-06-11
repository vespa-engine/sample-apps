package application;

import json.object.Album;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class Application implements Runnable {
    RandomAlbumGenerator albumGenerator;
    BlockingQueue<Album> queue;
    VespaDataFeeder dataFeeder;
    VespaQueryFeeder queryFeeder;
    private double pushProbability = 0.05;
    private double queryProbability = 0.05;
    private final int runsPerSecond = 100;
    private boolean isGrowing = true;
    private final Random random = new Random();
    private final AtomicInteger pendingQueryRequests;

    Application() {
        albumGenerator = new RandomAlbumGenerator();
        queue = new LinkedBlockingQueue<>();
        dataFeeder = new VespaDataFeeder(queue);
        dataFeeder.start();
        pendingQueryRequests = new AtomicInteger(0);
        queryFeeder = new VespaQueryFeeder(true, pendingQueryRequests);
        queryFeeder.start();
    }

    private void updatePushProbability() {
        pushProbability += isGrowing ? random.nextDouble() * 0.3 : - random.nextDouble() * 0.3;
        if(pushProbability >= 0.8) {
            pushProbability = 1.0;
            isGrowing = false;
        } else if (pushProbability <= 0.3) {
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

    public void run() {
        if (shouldPush()) queue.add(albumGenerator.getRandomAlbum());
        if (shouldQuery()) pendingQueryRequests.incrementAndGet();
    }

    public static void main(String[] args) {
        Timer timer = new Timer();
        Application app = new Application();

        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                app.run();
            }
        },
                0,
                (long) (1000.0 / (double) app.runsPerSecond));

        new Timer().scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                app.updatePushProbability();
                app.updateQueryProbability();
            }
        },
                0,
                10 * 1000);
    }
}

