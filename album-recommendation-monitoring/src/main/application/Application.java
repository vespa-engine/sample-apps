package application;

import java.io.IOException;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class Application implements Runnable {
    RandomAlbumGenerator albumGenerator = new RandomAlbumGenerator();
    DataPusher dataPusher = new DataPusher();
    private final int maxRequests = 55;
    private final int minRequests = 5;
    private final long timerDelay = 0;
    private final long timerPeriod = 1000;
    private boolean isGrowing = true;
    private int currentRequests = 5;
    private int runCounter = 1;
    private int pushCounter = 0;
    private Random random = new Random();

    private void updateRequestsPerMinute() {
        if (currentRequests >= 55) {
            isGrowing = false;
        } else if (currentRequests <= 5) {
            isGrowing = true;
        }
        currentRequests += isGrowing ? random.nextInt(5) : - random.nextInt(5);
    }

    private boolean shouldPush() {
        return ((double) pushCounter / (double) runCounter) <= (double) currentRequests / 60;
    }

    public void run() {
        if (shouldPush()) {
            pushNewAlbum();
            pushCounter++;
        }
        if(runCounter >= 60) {
            updateRequestsPerMinute();
            runCounter = 0;
            pushCounter = 0;
        }
        runCounter++;
    }

    public void pushNewAlbum() {
        dataPusher.pushThroughClass(albumGenerator.getRandomAlbum());
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
                app.timerDelay,
                app.timerPeriod);
    }
}

