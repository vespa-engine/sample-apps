package com.example;

import okhttp3.ConnectionPool;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Protocol;
import okhttp3.Request;
import okhttp3.Response;

import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
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
    };

    private static final AuthMethod AUTH_METHOD = AuthMethod.MTLS;

    private static final String ENDPOINT    = ""; 
    private static final String PUBLIC_CERT = "";
    private static final String PRIVATE_KEY = "";
    private static final String TOKEN       = "";

    private static final int LOAD_CONCURRENCY = 400;
    private static final int LOAD_NUM_QUERIES = 50000;

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
                    String result = runSingleQuery(createHttpClient(), "select * from sources * where userQuery()", query).get();
                    log.info(result);
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

    /**
     * Create a {@link OkHttpClient} for querying, with settings based on {@link VespaClient#AUTH_METHOD}.
     */
    static OkHttpClient createHttpClient() {
        var builder = new OkHttpClient.Builder()
            .protocols(Arrays.asList(Protocol.HTTP_2, Protocol.HTTP_1_1))
            .connectionPool(new ConnectionPool(200, 5, TimeUnit.MINUTES))
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(2, TimeUnit.SECONDS);

        switch (AUTH_METHOD) {
            case MTLS:
                {
                    var keyManager = PemUtils.loadIdentityMaterial(Path.of(PUBLIC_CERT), Path.of(PRIVATE_KEY));
                    var sslFactory = SSLFactory.builder()
                        .withIdentityMaterial(keyManager)
                        .withDefaultTrustMaterial()
                        .build();

                    builder.sslSocketFactory(sslFactory.getSslSocketFactory(), sslFactory.getTrustManager().orElseThrow());
                }
                break;
            case TOKEN:
                {
                    builder.addInterceptor(chain -> {
                        return chain.proceed(
                            chain.request()
                                 .newBuilder()
                                 .header("Authorization", "Bearer " + TOKEN)
                                 .build()
                        );
                    });
                }
                break;
            case NONE:
                break;
        }

        return builder.build();
    }

    /**
     * Create a {@link JsonFeeder} with settings based on {@link VespaClient#AUTH_METHOD}.
     */
    static JsonFeeder createFeeder() {
        FeedClientBuilder builder = FeedClientBuilder.create(URI.create(ENDPOINT));
        switch (AUTH_METHOD) {
            case MTLS:
                {
                    var keyManager = PemUtils.loadIdentityMaterial(
                            Path.of(PUBLIC_CERT),
                            Path.of(PRIVATE_KEY)
                            );
                    var sslFactory = SSLFactory.builder()
                        .withIdentityMaterial(keyManager)
                        .withDefaultTrustMaterial()
                        .build();

                    builder.setSslContext(sslFactory.getSslContext());
                }
                break;
            case TOKEN:
                {
                    builder.addRequestHeader("Authorization", "Bearer " + TOKEN);
                }
                break;
            case NONE:
                break;
        }
        FeedClient client = builder.build();
        return JsonFeeder.builder(client)
                         .withTimeout(Duration.ofSeconds(30))
                         .build();
    }

    static Optional<String> runSingleQuery(OkHttpClient client, String yql, String query) throws IOException {
        HttpUrl url = HttpUrl.parse(ENDPOINT + "search/")
            .newBuilder()
            .addQueryParameter("yql", yql)
            .addQueryParameter("query", query)
            .build();

        Request request = new Request.Builder()
            .url(url)
            .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.code() != 200) {
                throw  new IOException("Error code " + response.code());
            }
            if (response.body() != null) {
                // consume
                return Optional.of(response.body().string());
            }
        }
        return Optional.empty();
    }

    static void loadTest() throws Exception {
        var client = createHttpClient();

        ExecutorService executor = Executors.newFixedThreadPool(LOAD_CONCURRENCY);
        
        AtomicLong resultsReceived = new AtomicLong(0);
        AtomicLong errorsReceived = new AtomicLong(0);

        log.info("Performing " + LOAD_NUM_QUERIES + " queries with concurrency: " + LOAD_CONCURRENCY);

        long startTimeMillis = System.currentTimeMillis();

        for (int i = 0; i < LOAD_NUM_QUERIES; ++i) {
            executor.submit(() -> {
                try {
                    runSingleQuery(client, "select * from sources * where userQuery()", "guinness world record");
                } catch (Exception e) {
                    log.severe("Query iteration failed with: " + e.getMessage());
                    errorsReceived.incrementAndGet();
                } finally {
                    resultsReceived.incrementAndGet();
                }
            });
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        long timeSpentMillis = System.currentTimeMillis() - startTimeMillis;
        double qps = (double)(resultsReceived.get() - errorsReceived.get()) / (timeSpentMillis / 1000.0);
        log.info("----- Results -----");
        log.info("Received in total " + resultsReceived.get() + " results, " + errorsReceived.get() + " errors.");
        log.info("Time spent: " + timeSpentMillis + " ms.");
        log.info("QPS: " + qps);
    }

    /**
     * Feed documents from a .jsonl file given by {@code filePath}.
     */
    static void feedFromFile(String filePath) {
        try (FileInputStream jsonStream = new FileInputStream(filePath)) {
            JsonFeeder feeder = createFeeder();
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
            feeder.close();

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
