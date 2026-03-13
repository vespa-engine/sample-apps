package com.example;

import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpResponse.BodyHandler;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.help.HelpFormatter;

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
    private static final String PRIVATE_KEY = "/peth/to/private-key.pem";

    // Auth method: token.
    private static final String TOKEN       = "YOUR_TOKEN";

    // Number of concurrent in-flight HTTP/2 streams across all connections.
    private static final int    LOAD_POOL_SIZE   = 800;
    private static final int    LOAD_NUM_QUERIES = 1000000;
    // Each HttpClient opens its own connection. Multiple connections spread load
    // across container nodes via the load balancer.
    private static final int    NUM_CONNECTIONS  = 16;
    private static final String LOAD_TEST_YQL   = "select * from sources * where userQuery()";
    private static final String LOAD_TEST_QUERY = "guinness world record";

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
                try {
                    HttpResponse<String> response = runSingleQuery(createHttpClient(), "select * from sources * where userQuery()", query, HttpResponse.BodyHandlers.ofString()).get();
                    log.info(response.body());
                } catch (Exception e) {
                    log.severe("Query failed with message: " + e.getMessage());
                }
            } else {
                formatter.printHelp("VespaClient", "", options, "Error: No option specified", true);
            }
        } catch (ParseException e) {
            log.severe("Error parsing command line: " + e.getMessage());
            formatter.printHelp("VespaClient", "", options, "", true);
        }
    }

    static SSLFactory getSSLFactory() {
        var keyManager = PemUtils.loadIdentityMaterial(Path.of(PUBLIC_CERT), Path.of(PRIVATE_KEY));
        var sslFactory = SSLFactory.builder()
            .withIdentityMaterial(keyManager)
            .withDefaultTrustMaterial()
            .build();

        return sslFactory;
    }

    static HttpClient createHttpClient() {
        var clientBuilder = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_2)
            .connectTimeout(Duration.ofSeconds(5));

        if (AUTH_METHOD == AuthMethod.MTLS) {
            clientBuilder.sslContext(getSSLFactory().getSslContext());
        }

        return clientBuilder.build();
    }

    /**
     * Create a {@link JsonFeeder} with settings based on {@link VespaClient#AUTH_METHOD}.
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

    static <T> CompletableFuture<HttpResponse<T>> runSingleQuery(HttpClient client, String yql, String query, BodyHandler<T> handler) {
        String base = ENDPOINT.endsWith("/") ? ENDPOINT : ENDPOINT + "/";
        URI uri = URI.create(String.format("%ssearch/?yql=%s&query=%s",
            base,
            URLEncoder.encode(yql, StandardCharsets.UTF_8),
            URLEncoder.encode(query, StandardCharsets.UTF_8)));

        var reqBuilder = HttpRequest.newBuilder()
            .uri(uri)
            .GET()
            .timeout(Duration.ofSeconds(5));

        if (AUTH_METHOD == AuthMethod.TOKEN) {
            reqBuilder.header("Authorization", "Bearer " + TOKEN);
        }

        return client.sendAsync(reqBuilder.build(), handler);
    }

    static void loadTest() throws Exception {
        List<HttpClient> clients = new ArrayList<>(NUM_CONNECTIONS);
        for (int i = 0; i < NUM_CONNECTIONS; i++) {
            clients.add(createHttpClient());
        }

        log.info("Warmup: 100 synchronous queries");
        for (int i = 0; i < 100; ++i) {
            try {
                runSingleQuery(clients.get(i % NUM_CONNECTIONS), LOAD_TEST_YQL, LOAD_TEST_QUERY, HttpResponse.BodyHandlers.discarding()).get();
            } catch (Exception e) {
                log.severe("Warmup query failed: " + e.getMessage());
            }
        }

        log.info("Performing " + LOAD_NUM_QUERIES + " queries with " + LOAD_POOL_SIZE + " concurrent requests across " + NUM_CONNECTIONS + " connections");

        var remaining = new AtomicLong(LOAD_NUM_QUERIES);
        var resultsReceived = new AtomicLong(0);
        var errorsReceived = new AtomicLong(0);
        var latch = new CountDownLatch(LOAD_POOL_SIZE);

        long startTimeMillis = System.currentTimeMillis();

        for (int i = 0; i < LOAD_POOL_SIZE; i++) {
            sendNext(clients.get(i % NUM_CONNECTIONS), remaining, resultsReceived, errorsReceived, latch);
        }

        latch.await();

        long timeSpentMillis = System.currentTimeMillis() - startTimeMillis;
        double qps = (double)(resultsReceived.get() - errorsReceived.get()) / (timeSpentMillis / 1000.0);
        log.info("----- Results -----");
        log.info("Received in total " + resultsReceived.get() + " results, " + errorsReceived.get() + " errors.");
        log.info("Time spent: " + timeSpentMillis + " ms.");
        log.info("QPS: " + qps);
    }

    static void sendNext(HttpClient client, AtomicLong remaining,
                         AtomicLong resultsReceived, AtomicLong errorsReceived,
                         CountDownLatch latch) {
        if (remaining.decrementAndGet() < 0) {
            latch.countDown();
            return;
        }
        runSingleQuery(client, "select * from sources * where userQuery()",
                       "guinness world record", HttpResponse.BodyHandlers.discarding())
            .whenComplete((resp, ex) -> {
                if (ex != null) {
                    log.severe("Query failed: " + ex.getMessage());
                    errorsReceived.incrementAndGet();
                }
                resultsReceived.incrementAndGet();
                sendNext(client, remaining, resultsReceived, errorsReceived, latch);
            });
    }

    /**
     * Feed documents from a .jsonl file given by {@code filePath}.
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
