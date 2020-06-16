package application;

import com.google.gson.Gson;
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
import org.apache.http.conn.HttpHostConnectException;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.net.ConnectException;

public class VespaQueryFeeder extends Thread{

    CloseableHttpClient client;
    HttpPost post;
    private final Boolean debug;
    private final AtomicInteger pendingQueryRequests;
    private Boolean shouldRun = true;

    VespaQueryFeeder(Boolean debug, AtomicInteger pendingQueryRequests) {
        this.debug = debug;
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
            e.printStackTrace();
        }
    }

    public void queryVespa() {
        try {
            CloseableHttpResponse execute = client.execute(post);
            HttpEntity entity = execute.getEntity();

            if(debug && entity != null) {
                try (InputStream inputStream = entity.getContent()) {
                    String result = (new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))
                    .lines()
                    .collect(Collectors.joining("\n")));
                }
            }
        } catch (ConnectException | NoHttpResponseException e) {
            System.out.println("Unable to connect to vespa. Is it running?");
            try {
                Thread.sleep(10000);
            } catch (InterruptedException interruptedException) {
                interruptedException.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void shutDown() {
        shouldRun = false;
    }

    @Override
    public void run() {
        while(shouldRun) {
            if(pendingQueryRequests.get() > 0) {
                queryVespa();
                pendingQueryRequests.decrementAndGet();
            }
        }
    }

    public static void main(String[] args) {
        AtomicInteger count = new AtomicInteger(100);
        VespaQueryFeeder queryFeeder = new VespaQueryFeeder(true, count);
        queryFeeder.start();

    }
}
