package application;

import com.google.gson.Gson;
import java.io.IOException;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.atomic.AtomicInteger;
import json.ImmutableQuery;
import org.apache.http.NoHttpResponseException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class VespaQueryFeeder extends Thread {

    private final AtomicInteger pendingQueryRequests;
    Logger logger = LogManager.getLogger(VespaQueryFeeder.class);
    HttpClient client;
    HttpRequest request;
    private boolean shouldRun = true;

    VespaQueryFeeder(AtomicInteger pendingQueryRequests) {
        this.pendingQueryRequests = pendingQueryRequests;

        client = HttpClient.newHttpClient();
        ImmutableQuery query = ImmutableQuery.builder().yql("SELECT * FROM SOURCES * WHERE year > 2000;").build();

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
            logger.debug(result);
        } catch (ConnectException | NoHttpResponseException e) {
            logger.info("Unable to connect to vespa. Is it running?");
            Thread.sleep(10000);
        } catch (IOException e) {
            logger.error(e);
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
            logger.error(e);
            Thread.currentThread().interrupt();
        }
    }
}
