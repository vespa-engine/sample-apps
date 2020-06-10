package application;

import java.io.IOException;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

public class Application implements Runnable {
    RandomAlbumGenerator albumGenerator = new RandomAlbumGenerator();
    DataPusher dataPusher = new DataPusher();
    private final int maxRequests = 1200;
    private final int minRequests = 60;
    private final long timerDelay = 0;
    private final long timerPeriod = 10;
    private final int timeStep = 1000*10;
    private final int randomStepPerTimestep = (int) (100.0 * ((double) timeStep / (1000.0*60.0)));
    private boolean isGrowing = true;
    private int currentRequests = minRequests;
    private int runCounter = 1;
    private int pushCounter = 0;
    private final int runsPerTimestep = (int)(timeStep / timerPeriod);
    private final Random random = new Random();

    private void updateRequestsPerTimestep() {
        if (currentRequests >= maxRequests - randomStepPerTimestep) {
            isGrowing = false;
        } else if (currentRequests <= minRequests + randomStepPerTimestep) {
            isGrowing = true;
        }
        currentRequests += isGrowing ? random.nextInt(randomStepPerTimestep) : - random.nextInt(randomStepPerTimestep);
    }

    private boolean shouldPush() {
        return ((double) pushCounter / (double) runCounter) <= (double) currentRequests / runsPerTimestep;
    }

    public void run() {
        if (shouldPush()) {
            pushNewAlbum();
            pushCounter++;
        }
        if(runCounter >= runsPerTimestep) {
            updateRequestsPerTimestep();
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

