package com.example;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.ConnectionPool;
import okhttp3.Dispatcher;
import okhttp3.HttpUrl;
import okhttp3.OkHttpClient;
import okhttp3.Protocol;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
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

    private static final String ENDPOINT    = "https://eb1ee8bc.f6e11360.z.vespa-app.cloud/"; // TODO: change this
    private static final String PUBLIC_CERT = "/Users/magnuseifr/.vespa/vespa-team.example.magnuseifr/data-plane-public-cert.pem"; // TODO: change this
    private static final String PRIVATE_KEY = "/Users/magnuseifr/.vespa/vespa-team.example.magnuseifr/data-plane-private-key.pem"; // TODO: change this

    static void doQuery(String queryFilePath) throws Exception {
        int numConnections = 8;

        OkHttpClient[] clientPool = new OkHttpClient[numConnections];

        var keyManager = PemUtils.loadIdentityMaterial(Path.of(PUBLIC_CERT), Path.of(PRIVATE_KEY));
        var sslFactory = SSLFactory.builder()
            .withIdentityMaterial(keyManager)
            .withDefaultTrustMaterial()
            .build();


        for (int i = 0; i < numConnections; ++i) {
            var dispatcher = new Dispatcher();
            dispatcher.setMaxRequestsPerHost(100);
            dispatcher.setMaxRequests(100);
            clientPool[i] = new OkHttpClient.Builder()
                .dispatcher(dispatcher)
                .sslSocketFactory(sslFactory.getSslSocketFactory(), sslFactory.getTrustManager().orElseThrow())
                .protocols(Arrays.asList(Protocol.HTTP_2, Protocol.HTTP_1_1))
                .connectTimeout(5, TimeUnit.SECONDS)
                .readTimeout(10, TimeUnit.SECONDS)
                .build();
        }

        AtomicLong successes = new AtomicLong(0);
        AtomicLong errors = new AtomicLong(0);
        
        // Use a list or a counter to track total lines if you know it, 
        // or a dynamic approach. For a POC, let's assume we read the file once.
        long totalQueries = java.nio.file.Files.lines(Path.of(queryFilePath)).count();
        CountDownLatch latch = new CountDownLatch((int) totalQueries);

        long startTimeMillis = System.currentTimeMillis();

        int requestCounter = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader(queryFilePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    latch.countDown();
                    continue;
                }

                String[] parts = line.split(",");

                // 4. Use HttpUrl.Builder for cleaner, faster URL construction
                HttpUrl url = HttpUrl.parse(ENDPOINT + "search/")
                    .newBuilder()
                    .addQueryParameter("yql", parts[0])
                    .addQueryParameter("query", parts[1])
                    .build();

                Request request = new Request.Builder()
                    .url(url)
                    .build();

                var client = clientPool[requestCounter++ % numConnections];
                client.newCall(request).enqueue(new Callback() {
                    @Override
                    public void onFailure(Call call, IOException e) {
                        errors.incrementAndGet();
                        latch.countDown();
                    }

                    @Override
                    public void onResponse(Call call, Response response) {
                        // Crucial: The try-with-resources closes the response body, 
                        // which releases the HTTP/2 stream back to the pool.
                        try (ResponseBody body = response.body()) {
                            if (response.isSuccessful()) {
                                successes.incrementAndGet();
                            } else if (response.code() == 429 || response.code() >= 500) {
                                // Vespa: "Clients should reduce overall throughput 
                                // when receiving such responses."
                                System.err.println("Vespa is overloaded! Code: " + response.code());
                                errors.incrementAndGet();
                            } else {
                                errors.incrementAndGet();
                            }
                        } finally {
                            errors.incrementAndGet();
                            latch.countDown();
                        }
                    }
                });
            }
        }

        // 6. Block until all callbacks have called latch.countDown()
        latch.await();

        double secondsSpent = (double) (System.currentTimeMillis() - startTimeMillis) / 1000.0;
        System.out.println("--- Results ---");
        System.out.println("Success: " + successes.get() + ", Errors: " + errors.get());
        System.out.println("QPS: " + (successes.get() / secondsSpent));
        
        for (int i = 0; i < numConnections; ++i) {
            clientPool[i].dispatcher().executorService().shutdown();
        }
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
        options.addOption("q", "query", true, "Run queries");
        options.addOption("f", "feed", true, "Feed documents");

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter  = HelpFormatter.builder().get();

        try {
            CommandLine cmd = parser.parse(options, args);
            if (cmd.hasOption("q")) {
                String queryFilePath = cmd.getOptionValue("q");
                doQuery(queryFilePath);
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
