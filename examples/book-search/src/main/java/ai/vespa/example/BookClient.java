package ai.vespa.example;

import ai.vespa.feed.client.FeedClientBuilder;
import ai.vespa.feed.client.FeedException;
import ai.vespa.feed.client.JsonFeeder;
import ai.vespa.feed.client.Result;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

public class BookClient {

    private final String vespaUrl;
    private final HttpClient httpClient = HttpClient.newHttpClient();

    public BookClient(String vespaUrl) {
        this.vespaUrl = vespaUrl;
    }

    public FeedResult feed(Path documentsPath) throws IOException {
        try (var in         = Files.newInputStream(documentsPath);
             var feedClient = FeedClientBuilder.create(URI.create(vespaUrl)).build();
             var feeder     = JsonFeeder.builder(feedClient).build()) {

            var fed    = new AtomicLong(0);
            var errors = new AtomicLong(0);

            feeder.feedMany(in, new JsonFeeder.ResultCallback() {
                @Override public void onNextResult(Result result, FeedException error) {
                    fed.incrementAndGet();
                    if (error != null) errors.incrementAndGet();
                }
                @Override public void onError(FeedException error) {
                    errors.incrementAndGet();
                }
            }).join();

            return new FeedResult(fed.get(), errors.get());
        }
    }

    public void setLoanedOut(String docId, boolean loanedOut) throws IOException {
        var update = "{\"update\":\"id:library:book::" + docId + "\",\"fields\":{\"loaned_out\":{\"assign\":" + loanedOut + "}}}";
        try (var in         = new ByteArrayInputStream(update.getBytes(StandardCharsets.UTF_8));
             var feedClient = FeedClientBuilder.create(URI.create(vespaUrl)).build();
             var feeder     = JsonFeeder.builder(feedClient).build()) {
            feeder.feedMany(in, new JsonFeeder.ResultCallback() {
                @Override public void onNextResult(Result result, FeedException error) {
                    if (error != null) throw new RuntimeException("setLoanedOut failed: " + error.getMessage());
                }
                @Override public void onError(FeedException error) {
                    throw new RuntimeException("setLoanedOut failed: " + error.getMessage());
                }
            }).join();
        }
    }

    public String search(String yql) throws IOException, InterruptedException {
        return search(yql, null);
    }

    public String search(String yql, String query) throws IOException, InterruptedException {
        var url = new StringBuilder(vespaUrl)
                .append("/search/?yql=").append(URLEncoder.encode(yql, StandardCharsets.UTF_8));
        if (query != null)
            url.append("&query=").append(URLEncoder.encode(query, StandardCharsets.UTF_8));

        var request  = HttpRequest.newBuilder().uri(URI.create(url.toString())).GET().build();
        var response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public record FeedResult(long fed, long errors) {}

}
