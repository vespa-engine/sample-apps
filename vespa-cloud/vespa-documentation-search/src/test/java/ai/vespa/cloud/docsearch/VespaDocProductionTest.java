package ai.vespa.cloud.docsearch;

import ai.vespa.hosted.cd.ProductionTest;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ProductionTest
public class VespaDocProductionTest {

    HttpClient httpClient = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_2)
            .build();

    @Test
    void verifyMetrics() throws IOException, InterruptedException {

        // Can use publicly available resources only, so use open query interface
        // Here, ensure > 50 documents about ranking
        HttpRequest req = HttpRequest.newBuilder()
                .GET()
                .uri(URI.create("https://doc-search.vespa.oath.cloud/search/?query=ranking&ranking=documentation&locale=en-US&hits=1"))
                .build();
        HttpResponse<String> res = httpClient.send(req, HttpResponse.BodyHandlers.ofString());

        assertEquals(200, res.statusCode());

        String body = res.body();
        long hitCount = new ObjectMapper().readTree(body)
                .get("root").get("fields").get("totalCount").asLong();
        assert(hitCount > 50);
    }
}
