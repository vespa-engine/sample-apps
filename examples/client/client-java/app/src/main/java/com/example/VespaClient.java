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

import ai.vespa.feed.client.FeedClientBuilder;
import nl.altindag.ssl.SSLFactory;
import nl.altindag.ssl.pem.util.PemUtils;

public class VespaClient {
    private final static Logger log = Logger.getLogger(VespaClient.class.getName());

    private enum AuthMethod {
        MTLS,   // mTLS: Recommended for Vespa Cloud
        TOKEN,  // Token-based authentication
        NONE    // E.g. if self-hosting.
    };

    private static final AuthMethod AUTH_METHOD = AuthMethod.TOKEN;

    private static final String ENDPOINT    = ""; // TODO: change this
    private static final String PUBLIC_CERT = ""; // TODO: change this
    private static final String PRIVATE_KEY = ""; // TODO: change this
    private static final String TOKEN       = "";

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
     *
     */
    static JsonFileFeeder createFeeder() {
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
        return new JsonFileFeeder(builder.build());
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
        int concurrency = 400; // Scale until system reaches limit
        int totalQueries = 50000;

        var client = createHttpClient();

        ExecutorService executor = Executors.newFixedThreadPool(concurrency);
        
        AtomicLong successes = new AtomicLong(0);
        AtomicLong errors = new AtomicLong(0);

        log.info("Performing " + totalQueries + " queries with concurrency: " + concurrency);

        long startTimeMillis = System.currentTimeMillis();

        for (int i = 0; i < totalQueries; ++i) {
            executor.submit(() -> {
                try {
                    runSingleQuery(client, "select * from sources * where userQuery()", "guinness world record");
                    successes.incrementAndGet();
                } catch (Exception e) {
                    log.severe("Iteration failed with: " + e.getMessage());
                    errors.incrementAndGet();
                }
            });
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        double secondsSpent = (double) (System.currentTimeMillis() - startTimeMillis) / 1000.0;
        log.info("----- Results -----");
        log.info("Success: " + successes.get() + ", Errors: " + errors.get());
        log.info("QPS: " + (successes.get() / secondsSpent));
    }

    /**
     * Feed documents from a .jsonl file given by feedPath.
     */
    static void feedFromFile(String filePath) {
        try (FileInputStream is = new FileInputStream(filePath)) {
            JsonFileFeeder feeder = createFeeder();
            feeder.batchFeed(is);
            feeder.close();
        } catch (IOException e) {
            log.severe("Error when trying to feed documents: " + e.getMessage());
        }
    }

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
}
