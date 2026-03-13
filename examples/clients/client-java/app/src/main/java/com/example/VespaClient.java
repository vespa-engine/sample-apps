/*
 * Copyright Vespa.ai. All rights reserved.
 */
package com.example;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.channels.AsynchronousCloseException;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.help.HelpFormatter;

import org.eclipse.jetty.client.ContentResponse;
import org.eclipse.jetty.client.HttpClient;
import org.eclipse.jetty.client.RetryableRequestException;
import org.eclipse.jetty.http.HttpHeader;
import org.eclipse.jetty.http.HttpMethod;
import org.eclipse.jetty.http2.client.HTTP2Client;
import org.eclipse.jetty.http2.client.transport.HttpClientTransportOverHTTP2;
import org.eclipse.jetty.io.ClientConnector;
import org.eclipse.jetty.util.ssl.SslContextFactory;

import ai.vespa.feed.client.DocumentId;
import ai.vespa.feed.client.FeedClient;
import ai.vespa.feed.client.FeedClientBuilder;
import ai.vespa.feed.client.FeedException;
import ai.vespa.feed.client.JsonFeeder;
import ai.vespa.feed.client.Result;
import nl.altindag.ssl.SSLFactory;
import nl.altindag.ssl.pem.util.PemUtils;


public class VespaClient {
    private final static Logger log = Logger.getLogger(VespaClient.class.getName());

    private enum AuthMethod {
        MTLS,   // mTLS: Recommended for Vespa Cloud
        TOKEN,  // Token-based authentication
        NONE    // E.g. if self-hosting.
    }

    private static final AuthMethod AUTH_METHOD = AuthMethod.MTLS;

    private static final String ENDPOINT    = "YOUR_ENDPOINT";
    // Auth method: mTLS
    private static final String PUBLIC_CERT = "/path/to/public-cert.pem";
    private static final String PRIVATE_KEY = "/path/to/private-key.pem";
    // Auth method: token.
    private static final String TOKEN       = "YOUR_TOKEN";

    // Number of concurrent in-flight HTTP/2 streams across all connections.
    private static final int    CONCURRENT_REQUESTS = 512;
    private static final int    TOTAL_QUERIES       = 1000000;
    // Max connections to keep open at the same time. Excess requests are queued on existing
    // connections rather than opening new ones, preventing connection explosion under load.
    private static final int    NUM_CONNECTIONS     = 8;
    private static final String QUERY_YQL           = "select * from sources * where userQuery()";
    private static final String QUERY_INPUT         = "guinness world record";

    /**
     * Parses command-line arguments and dispatches to the appropriate operation:
     * single query ({@code -q}), load test ({@code -l}), or document feed ({@code -f}).
     */
    public static void main(String[] args) throws Exception {
        Options options = new Options();
        options.addOption("q", "query", true, "Run one query");
        options.addOption("l", "load-test", false, "Run many queries");
        options.addOption("f", "feed", true, "Feed documents");

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter  = HelpFormatter.builder().get();

        try {
            CommandLine cmd = parser.parse(options, args);
            if (cmd.hasOption("l")) {
                loadTest();
            } else if (cmd.hasOption("f")) {
                String feedPath = cmd.getOptionValue("f");
                feedFromFile(feedPath);
            } else if (cmd.hasOption("q")) {
                String query = cmd.getOptionValue("q");
                var client = createHttpClient();
                try {
                    String base = ENDPOINT.endsWith("/") ? ENDPOINT : ENDPOINT + "/";
                    String uri = String.format("%ssearch/?yql=%s&query=%s",
                        base,
                        URLEncoder.encode(QUERY_YQL, StandardCharsets.UTF_8),
                        URLEncoder.encode(query, StandardCharsets.UTF_8));
                    var request = client.newRequest(uri).timeout(5, TimeUnit.SECONDS);
                    if (AUTH_METHOD == AuthMethod.TOKEN) {
                        request.headers(h -> h.add(HttpHeader.AUTHORIZATION, "Bearer " + TOKEN));
                    }
                    ContentResponse response = request.send();
                    log.info(response.getContentAsString());
                } catch (Exception e) {
                    log.severe("Query failed with message: " + e.getMessage());
                } finally {
                    client.stop();
                }
            } else {
                formatter.printHelp("VespaClient", "", options, "Error: No option specified", true);
            }
        } catch (ParseException e) {
            log.severe("Error parsing command line: " + e.getMessage());
            formatter.printHelp("VespaClient", "", options, "", true);
        }
    }

    /**
     * Builds an {@link SSLFactory} loaded with the mTLS client certificate and private key.
     */
    static SSLFactory getSSLFactory() {
        var keyManager = PemUtils.loadIdentityMaterial(Path.of(PUBLIC_CERT), Path.of(PRIVATE_KEY));
        var sslFactory = SSLFactory.builder()
            .withIdentityMaterial(keyManager)
            .withDefaultTrustMaterial()
            .build();

        return sslFactory;
    }

    /**
     * Creates a Jetty {@link HttpClient} configured for HTTP/2 and the selected {@link AuthMethod}.
     * Caps connections at {@link #NUM_CONNECTIONS} per destination and queues excess requests
     * rather than opening new connections, preventing connection explosion under high concurrency.
     */
    static HttpClient createHttpClient() throws Exception {
        var ssl = new SslContextFactory.Client();
        if (AUTH_METHOD == AuthMethod.MTLS) {
            ssl.setSslContext(getSSLFactory().getSslContext());
        }

        var connector = new ClientConnector();
        connector.setSslContextFactory(ssl);

        var client = new HttpClient(new HttpClientTransportOverHTTP2(new HTTP2Client(connector)));
        client.setMaxConnectionsPerDestination(NUM_CONNECTIONS);
        client.setMaxRequestsQueuedPerDestination(CONCURRENT_REQUESTS);
        client.start();

        return client;
    }

    /**
     * Creates a {@link JsonFeeder} configured for the selected {@link AuthMethod}.
     */
    static JsonFeeder createFeeder() {
        FeedClientBuilder builder = FeedClientBuilder.create(URI.create(ENDPOINT));
        switch (AUTH_METHOD) {
            case MTLS:
                builder.setSslContext(getSSLFactory().getSslContext());
                break;
            case TOKEN:
                builder.addRequestHeader("Authorization", "Bearer " + TOKEN);
                break;
            case NONE:
                break;
        }
        FeedClient client = builder.build();
        return JsonFeeder.builder(client)
                         .withTimeout(Duration.ofSeconds(30))
                         .build();
    }

    /**
     * Sends a single search query asynchronously.
     *
     * @param client the HTTP client to use
     * @param yql    the YQL query string
     * @param query  the user query input passed to {@code userQuery()}
     */
    static CompletableFuture<Void> runSingleQuery(HttpClient client, String yql, String query) {
        String base = ENDPOINT.endsWith("/") ? ENDPOINT : ENDPOINT + "/";
        String uri = String.format("%ssearch/?yql=%s&query=%s",
            base,
            URLEncoder.encode(yql, StandardCharsets.UTF_8),
            URLEncoder.encode(query, StandardCharsets.UTF_8));

        var request = client.newRequest(uri)
            .method(HttpMethod.GET)
            .timeout(30, TimeUnit.SECONDS);

        if (AUTH_METHOD == AuthMethod.TOKEN) {
            request.headers(h -> h.add(HttpHeader.AUTHORIZATION, "Bearer " + TOKEN));
        }

        var cf = new CompletableFuture<Void>();
        request.send(result -> {
            if (result.isFailed()) cf.completeExceptionally(result.getFailure());
            else cf.complete(null);
        });
        return cf;
    }

    /**
     * Runs a load test against the search endpoint, reporting QPS on completion.
     * Tune {@link #CONCURRENT_REQUESTS}, {@link #TOTAL_QUERIES}, and {@link #NUM_CONNECTIONS} to adjust load.
     */
    static void loadTest() throws Exception {
        var client = createHttpClient();
        try {
            // Warmup: one request per connection to ensure all NUM_CONNECTIONS TLS handshakes
            // complete before the timed test begins, giving a clean steady-state QPS measurement.
            log.info("Warmup: establishing " + NUM_CONNECTIONS + " connections");
            var warmupLatch = new CountDownLatch(NUM_CONNECTIONS);
            for (int i = 0; i < NUM_CONNECTIONS; i++) {
                runWithRetry(client, QUERY_YQL, QUERY_INPUT, 3)
                    .whenComplete((v, ex) -> {
                        if (ex != null) log.severe("Warmup query failed: " + ex.getMessage());
                        warmupLatch.countDown();
                    });
            }
            warmupLatch.await();

            log.info("Performing " + TOTAL_QUERIES + " queries with " + CONCURRENT_REQUESTS + " concurrent requests across " + NUM_CONNECTIONS + " connections");

            var remaining = new AtomicLong(TOTAL_QUERIES);
            var resultsReceived = new AtomicLong(0);
            var errorsReceived = new AtomicLong(0);
            var latch = new CountDownLatch(CONCURRENT_REQUESTS);

            long startTimeMillis = System.currentTimeMillis();

            for (int i = 0; i < CONCURRENT_REQUESTS; i++) {
                sendNext(client, remaining, resultsReceived, errorsReceived, latch);
            }

            latch.await();

            long timeSpentMillis = System.currentTimeMillis() - startTimeMillis;
            double qps = (double)(resultsReceived.get() - errorsReceived.get()) / (timeSpentMillis / 1000.0);
            log.info("----- Results -----");
            log.info("Received in total " + resultsReceived.get() + " results, " + errorsReceived.get() + " errors.");
            log.info("Time spent: " + timeSpentMillis + " ms.");
            log.info("QPS: " + qps);
        } finally {
            client.stop();
        }
    }

    /**
     * Fires the next query asynchronously and chains itself as the completion callback,
     * keeping exactly {@link #CONCURRENT_REQUESTS} requests in flight without blocking any threads.
     * Signals {@code latch} when this slot's share of queries is exhausted.
     */
    static void sendNext(HttpClient client, AtomicLong remaining,
                         AtomicLong resultsReceived, AtomicLong errorsReceived,
                         CountDownLatch latch) {
        if (remaining.decrementAndGet() < 0) {
            latch.countDown();
            return;
        }
        runWithRetry(client, QUERY_YQL, QUERY_INPUT, 3)
            .whenComplete((v, ex) -> {
                if (ex != null) {
                    errorsReceived.incrementAndGet();
                    log.severe("Query failed: " + ex.getMessage() + " (" + ex.getClass().getSimpleName() + ")");
                }
                resultsReceived.incrementAndGet();
                sendNext(client, remaining, resultsReceived, errorsReceived, latch);
            });
    }

    /**
     * Sends a query, transparently retrying up to {@code retriesLeft} times on
     * {@link RetryableRequestException} — typically caused by server-side connection
     * recycling (GOAWAY) when Jetty's internal retry queue is exhausted.
     */
    static CompletableFuture<Void> runWithRetry(HttpClient client, String yql, String query, int retriesLeft) {
        return runSingleQuery(client, yql, query)
            .exceptionallyCompose(ex -> {
                if ((ex instanceof RetryableRequestException || ex instanceof AsynchronousCloseException) && retriesLeft > 0) {
                    log.warning("Retrying transient error (" + retriesLeft + " attempts left): " + ex.getMessage());
                    return runWithRetry(client, yql, query, retriesLeft - 1);
                }
                return CompletableFuture.failedFuture(ex);
            });
    }

    /**
     * Feeds documents from a JSON Lines file at {@code filePath} using the Vespa feed client,
     * reporting feed rate on completion.
     */
    static void feedFromFile(String filePath) {
        try (FileInputStream jsonStream = new FileInputStream(filePath);
             JsonFeeder feeder = createFeeder()) {
            log.info("Starting feed");

            AtomicLong resultsReceived = new AtomicLong(0);
            AtomicLong errorsReceived = new AtomicLong(0);

            long startTimeMillis = System.currentTimeMillis();

            var promise = feeder.feedMany(jsonStream, new JsonFeeder.ResultCallback() {
                @Override
                public void onNextResult(Result result, FeedException error) {
                    resultsReceived.incrementAndGet();
                    if (error != null) {
                        log.warning("Problems with feeding document "
                            + error.documentId().map(DocumentId::toString).orElse("<unknown>")
                            + ": " + error
                        );
                        errorsReceived.incrementAndGet();
                    }
                }

                @Override
                public void onError(FeedException error) {
                    log.severe("Feeding failed fatally: " + error.getMessage());
                }
            });

            promise.join();

            long timeSpentMillis = (System.currentTimeMillis() - startTimeMillis);
            double okRatePerSec = (double)(resultsReceived.get() - errorsReceived.get()) / (timeSpentMillis / 1000.0);
            log.info("----- Results ----");
            log.info("Received in total " + resultsReceived.get() + " results, " + errorsReceived.get() + " errors.");
            log.info("Time spent: " + timeSpentMillis + " ms.");
            log.info("OK-rate: " + okRatePerSec + "/s");
        } catch (IOException e) {
            log.severe("Fatal error when trying to feed documents: " + e.getMessage());
        }
    }
}
