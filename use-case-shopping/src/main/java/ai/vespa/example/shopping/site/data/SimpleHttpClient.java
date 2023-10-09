// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.data;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.io.IOUtils;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.client.methods.HttpUriRequest;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;

import java.io.Closeable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Optional;

public class SimpleHttpClient implements Closeable {

    private final CloseableHttpClient httpClient;
    private final String host;

    public SimpleHttpClient(String protocol, String hostName, String hostPort) {
        httpClient = HttpClientBuilder.create().build();
        this.host = String.format("%s://%s:%s", protocol, hostName, hostPort);
    }

    public String get(String path) throws IOException {
        HttpGet httpGet = new HttpGet(host + path);
        return execute(httpGet);
    }

    public String put(String path, String payload) throws IOException {
        HttpPut httpPut = new HttpPut(host + path);
        httpPut.setEntity(new StringEntity(payload));
        return execute(httpPut);
    }

    private String execute(HttpUriRequest req) throws IOException {
        HttpResponse httpResponse = httpClient.execute(req);
        HttpEntity entity = httpResponse.getEntity();
        String result = IOUtils.readAll(entity.getContent(), StandardCharsets.UTF_8);
        EntityUtils.consume(entity);
        if (httpResponse.getStatusLine().getStatusCode() != 200) {
            return null;
        }
        return result;
    }

    @Override
    public void close() throws IOException {
        httpClient.close();
    }

    public JsonNode getJson(String url) {
        if (url == null) {
            return null;
        }
        try {
            String response = get(url);
            if (response == null) {
                throw new RuntimeException("No results returned for query '" + url + "'");
            }
            return new ObjectMapper().readTree(response);
        } catch (IOException e) {
            throw new RuntimeException("Error handling query for url '" + url + "'", e);
        }
    }


    public static Optional<JsonNode> getNode(JsonNode node, String... path) {
        if (node == null) {
            return Optional.empty();
        }
        if (path.length == 0) {
            return Optional.of(node);
        }
        for (String p : path) {
            node = node.get(p);
            if (node == null) {
                return Optional.empty();
            }
            if (node.isArray()) {
                node = node.get(0);  // use first element if an array
            }
        }
        return Optional.of(node);
    }

    public static Optional<String> getStringValue(JsonNode root, String... path) {
        return getNode(root, path).map(JsonNode::asText);
    }

    public static Optional<Integer> getIntValue(JsonNode root, String... path) {
        return getNode(root, path).map(JsonNode::asInt);
    }

    public static Optional<Double> getDoubleValue(JsonNode root, String... path) {
        return getNode(root, path).map(JsonNode::asDouble);
    }

}
