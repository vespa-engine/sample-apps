package application;

import com.google.gson.Gson;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.ConnectException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import json.ImmutableQuery;
import org.apache.http.HttpEntity;
import org.apache.http.NameValuePair;
import org.apache.http.NoHttpResponseException;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class VespaQueryFeeder extends Thread {

    private final AtomicInteger pendingQueryRequests;
    Logger logger = LogManager.getLogger(VespaQueryFeeder.class);
    CloseableHttpClient client;
    HttpPost post;
    private boolean shouldRun = true;

    VespaQueryFeeder(AtomicInteger pendingQueryRequests) {
        this.pendingQueryRequests = pendingQueryRequests;

        client = HttpClients.createDefault();
        post = new HttpPost("http://vespa:8080/search/");

        post.setHeader("Content-Type", "java/application/java.json");

        List<NameValuePair> params = new ArrayList<>();

        ImmutableQuery query = ImmutableQuery.builder().yql("select * from sources * where year > 2000;").build();

        try {
            post.setEntity(new UrlEncodedFormEntity(params, "UTF-8"));
            post.setEntity(new StringEntity(new Gson().toJson(query)));
        } catch (UnsupportedEncodingException e) {
            logger.error(e);
        }
    }

    public void queryVespa() throws InterruptedException {
        try {
            CloseableHttpResponse execute = client.execute(post);
            HttpEntity entity = execute.getEntity();
            if (entity != null) {
                InputStream inputStream = entity.getContent();
                logger.log(Level.DEBUG, () -> new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))
                        .lines()
                        .collect(Collectors.joining("\n")));
            }
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
