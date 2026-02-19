package com.example;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;

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

    private static final String ENDPOINT    = "https://vespa-endpoint";
    private static final String PUBLIC_CERT = "/path/to/data-plane-public-cert.pem";
    private static final String PRIVATE_KEY = "/path/to/data-plane-private-key.pem";

    static OkHttpClient createClient() {
        var keyManager = PemUtils.loadIdentityMaterial(
            Path.of(PUBLIC_CERT),
            Path.of(PRIVATE_KEY)
        );
        var sslFactory = SSLFactory.builder()
            .withIdentityMaterial(keyManager)
            .withDefaultTrustMaterial()
            .build();
        return new OkHttpClient.Builder()
            .sslSocketFactory(sslFactory.getSslSocketFactory(), sslFactory.getTrustManager().orElseThrow())
            .build();
    }

    static void doQuery() throws IOException {
        var client = createClient();

        var request = new Request.Builder()
            .url(ENDPOINT + "search/?yql=select%20*%20from%20sources%20*%20where%20true")
            .build();

        try (Response response = client.newCall(request).execute()) {
            System.out.println("Status: " + response.code());
            System.out.println("Body: " + response.body().string());
        }
    }

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
        options.addOption("q", "query", false, "Run a query");
        options.addOption("f", "feed", true, "Feed documents");

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter  = HelpFormatter.builder().get();

        try {
            CommandLine cmd = parser.parse(options, args);
            if (cmd.hasOption("q")) {
                doQuery();
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
