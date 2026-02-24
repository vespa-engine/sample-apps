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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.help.HelpFormatter;

import nl.altindag.ssl.SSLFactory;
import nl.altindag.ssl.pem.util.PemUtils;

public class VespaClient {

    private enum AuthMethod {
        MTLS,   // Recommended for Vespa Cloud
        TOKEN,  // Token header authentication
        NONE    // E.g. if self-hosting.
    };

    private static final AuthMethod AUTH_METHOD = AuthMethod.MTLS;

    private static final String ENDPOINT    = ""; // TODO: change this
    private static final String PUBLIC_CERT = ""; // TODO: change this
    private static final String PRIVATE_KEY = ""; // TODO: change this

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
                    // TODO
                }
                break;
            case NONE:
                {
                }
                break;
        }

        return builder.build();
    }

    static void runSingleQuery(OkHttpClient client, String yql, String query) throws IOException {
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
                response.body().string();
            }
        }
    }

    static void doLoadTest() throws Exception {
        int concurrency = 400; // Scale until system reaches limit
        int totalQueries = 50000;

        var client = createHttpClient();

        ExecutorService executor = Executors.newFixedThreadPool(concurrency);
        
        AtomicLong successes = new AtomicLong(0);
        AtomicLong errors = new AtomicLong(0);

        long startTimeMillis = System.currentTimeMillis();

        for (int i = 0; i < totalQueries; ++i) {
            executor.submit(() -> {
                try {
                    runSingleQuery(client, "select * from sources * where userQuery()", "common words here?");
                    successes.incrementAndGet();
                } catch (Exception e) {
                    System.err.println("Iteration failed with: " + e.getMessage());
                    errors.incrementAndGet();
                }
            });
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.HOURS);

        double secondsSpent = (double) (System.currentTimeMillis() - startTimeMillis) / 1000.0;
        System.out.println("--- Results ---");
        System.out.println("Success: " + successes.get() + ", Errors: " + errors.get());
        System.out.println("QPS: " + (successes.get() / secondsSpent));
    }

    /**
     * Feed documents from a .jsonl file given by feedPath.
     */
    static void doFeed(String feedPath) {
        try (FileInputStream is = new FileInputStream(feedPath)) {
            var keyManager = PemUtils.loadIdentityMaterial(
                Path.of(PUBLIC_CERT),
                Path.of(PRIVATE_KEY)
            );
            var sslFactory = SSLFactory.builder()
                .withIdentityMaterial(keyManager)
                .withDefaultTrustMaterial()
                .build();

            var feeder = new JsonFileFeeder(URI.create(ENDPOINT), sslFactory.getSslContext());
            feeder.batchFeed(is, "batchId");

            feeder.close();
        } catch (IOException e) {
            System.err.println("Error when trying to feed documents: " + e.getMessage());
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
                doLoadTest();
            } else if (cmd.hasOption("f")) {
                String feedPath = cmd.getOptionValue("f");
                doFeed(feedPath);
            } else {
                formatter.printHelp("VespaClient", "", options, "Error: No option specified", true);
            }
        } catch (ParseException e) {
            System.err.println("Error parsing command line: " + e.getMessage());
            formatter.printHelp("VespaClient", "", options, "", true);
        }
    }
}
