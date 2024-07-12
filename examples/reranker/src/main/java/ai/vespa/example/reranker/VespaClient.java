// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.reranker;

import com.yahoo.container.jdisc.HttpRequest;
import org.apache.hc.client5.http.classic.methods.HttpGet;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClientBuilder;
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManagerBuilder;
import org.apache.hc.core5.http.ClassicHttpResponse;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.HttpClientResponseHandler;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.apache.hc.core5.net.URIBuilder;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Map;

/**
 * A client which can talk to a Vespa applications *token* endpoint.
 * This is multithread safe.
 *
 * @author bratseth
 */
class VespaClient {

    private final String tokenEndpoint;
    private final CloseableHttpClient httpClient;

    public VespaClient(String tokenEndpoint) {
        this.tokenEndpoint = tokenEndpoint;
        this.httpClient = HttpClientBuilder.create()
                                           .setConnectionManager(PoolingHttpClientConnectionManagerBuilder
                                                                         .create()
                                                                         .build())
                                           .setUserAgent("vespa")
                                           .disableCookieManagement()
                                           .disableAutomaticRetries()
                                           .disableAuthCaching()
                                           .build();
    }

    public Response search(HttpRequest request, Map<String, Object> overridingProperties) throws IOException {
        try {
            String authorizationHeader = request.getHeader("Authorization");
            if (authorizationHeader == null || !authorizationHeader.startsWith("Bearer "))
                throw new IllegalArgumentException("Request must have an 'Authorization' header with the value " +
                                                   "'Bearer $your_token'");
            // String tokenHc = "vespa_cloud_dNpDIa7RkNntm0AkvKWNlA0cFydFa4W3GlV6HOGQTuf";
            // String authorizationHeader = "Bearer " + authorizationHeader;
            var uriBuilder = new URIBuilder(tokenEndpoint);
            uriBuilder.setPath("/search/");
            for (var property : request.propertyMap().entrySet())
                uriBuilder.addParameter(property.getKey(), property.getValue());
            for (var property : overridingProperties.entrySet())
                uriBuilder.addParameter(property.getKey(), property.getValue().toString());
            var get = new HttpGet(uriBuilder.build());
            get.addHeader("Authorization", authorizationHeader);
            return httpClient.execute(get, new ResponseHandler());
        }
        catch (URISyntaxException e) {
            throw new IllegalStateException(e);
        }
    }

    public record Response(int statusCode, String responseBody) {}

    // Custom ResponseHandler to handle the response
    public static class ResponseHandler implements HttpClientResponseHandler<Response> {

        @Override
        public Response handleResponse(ClassicHttpResponse response) {
            String responseBody;
            try {
                responseBody = EntityUtils.toString(response.getEntity());
            } catch (IOException | ParseException e) {
                throw new IllegalStateException(e);
            }
            return new Response(response.getCode(), responseBody);
        }
    }

}
