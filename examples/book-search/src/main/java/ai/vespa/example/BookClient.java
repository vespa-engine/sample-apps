package ai.vespa.example;

import ai.vespa.feed.client.FeedClientBuilder;
import ai.vespa.feed.client.FeedException;
import ai.vespa.feed.client.JsonFeeder;
import ai.vespa.feed.client.Result;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple client for feeding and querying the book search application.
 */
public class BookClient {

    private static final String NAMESPACE = "library";
    private static final String SCHEMA    = "book";

    private final String     vespaUrl;
    private final HttpClient httpClient = HttpClient.newHttpClient();

    public BookClient(String vespaUrl) {
        this.vespaUrl = vespaUrl;
    }

    /** Feeds documents from the given path, returning the number of successfully fed documents and errors. */
    public FeedResult feed(Path documentsPath) throws IOException {
        try (InputStream in    = Files.newInputStream(documentsPath);
             JsonFeeder feeder = newFeeder()) {

            AtomicLong fed    = new AtomicLong(0);
            AtomicLong errors = new AtomicLong(0);

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

    /** Updates the 'loaned_out' field of the given document. */
    public void setLoanedOut(String docId, boolean loanedOut) throws IOException {
        String update = "{\"update\":\"id:" + NAMESPACE + ":" + SCHEMA + "::" + docId + "\","
                      + "\"fields\":{\"loaned_out\":{\"assign\":" + loanedOut + "}}}";
        try (JsonFeeder feeder = newFeeder()) {
            feeder.feedMany(new ByteArrayInputStream(update.getBytes(StandardCharsets.UTF_8)),
                    new JsonFeeder.ResultCallback() {
                        @Override public void onNextResult(Result result, FeedException error) {
                            if (error != null) throw new RuntimeException("setLoanedOut failed: " + error.getMessage());
                        }
                        @Override public void onError(FeedException error) {
                            throw new RuntimeException("setLoanedOut failed: " + error.getMessage());
                        }
                    }).join();
        }
    }

    /** Executes the given YQL query, returning the raw JSON response as a string. */
    public String query(String yql) throws IOException, InterruptedException {
        return query(yql, null);
    }

    /** Executes the given YQL query with an optional user query, returning the raw JSON response as a string. */
    public String query(String yql, String userQuery) throws IOException, InterruptedException {
        StringBuilder url = new StringBuilder(vespaUrl)
                .append("/search/?yql=").append(URLEncoder.encode(yql, StandardCharsets.UTF_8));
        if (userQuery != null)
            url.append("&query=").append(URLEncoder.encode(userQuery, StandardCharsets.UTF_8));

        HttpRequest request = HttpRequest.newBuilder().uri(URI.create(url.toString())).GET().build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    public record FeedResult(long fed, long errors) {}

    // --- Helpers ---

    private JsonFeeder newFeeder() {
        return JsonFeeder.builder(FeedClientBuilder.create(URI.create(vespaUrl)).build()).build();
    }
}
