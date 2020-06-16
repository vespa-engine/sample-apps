package application;

import com.google.gson.Gson;
import com.yahoo.vespa.http.client.FeedClient;
import com.yahoo.vespa.http.client.FeedClientFactory;
import com.yahoo.vespa.http.client.SimpleLoggerResultCallback;
import com.yahoo.vespa.http.client.config.*;
import json.Album;
import json.ImmutableTopLevelPut;
import json.TopLevelPut;

import java.net.ConnectException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

public class VespaDataFeeder extends Thread{

    private final FeedClient feedClient;
    private final Gson gson = new Gson();
    private AtomicInteger pending = new AtomicInteger(0);
    private final BlockingQueue<Album> queue;
    private final String idFormat = "id:mynamespace:music::%d";
    private boolean shouldRun = true;


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
                System.out.println("Got an exception");
                break;
            }
            String id = String.format(idFormat, pending.incrementAndGet());
            TopLevelPut newAlbum = ImmutableTopLevelPut.builder()
                    .fields(album)
                    .put(id)
                    .build();
            feedClient.stream(id, gson.toJson(newAlbum));
        }
    }
}
