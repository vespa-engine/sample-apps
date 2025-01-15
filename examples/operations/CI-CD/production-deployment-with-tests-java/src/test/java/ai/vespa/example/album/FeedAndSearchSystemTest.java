// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.SystemTest;
import ai.vespa.hosted.cd.TestRuntime;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestReporter;

import java.io.IOException;
import java.net.http.HttpResponse;
import java.util.Map;

import static java.net.http.HttpRequest.BodyPublishers.ofString;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SystemTest
class FeedAndSearchSystemTest {

    private final Endpoint endpoint = TestRuntime.get().deploymentToTest().endpoint("default");

    @Test
    void testOutput(TestReporter testReporter) {
        testReporter.publishEntry("Hello from an empty test!");
        assertTrue(true, "Text from assertion for comparison");
    }

    @Test
    void feedAndSearch() throws IOException {
        String documentPath = "/document/v1/mynamespace/music/docid/test1";
        String deleteAll = "/document/v1/mynamespace/music/docid?selection=true&cluster=music";
        String document = """
            {
                "fields": {
                    "artist": "Coldplay"
                }
            }""";
        String yql = "SELECT * FROM music WHERE artist CONTAINS 'coldplay'";

        HttpResponse<String> deleteResult = endpoint.send(endpoint.request(deleteAll).DELETE());
        assertEquals(200, deleteResult.statusCode());

        // the first query needs a higher timeout than the default 500ms, to warm up the code
        HttpResponse<String> emptyResult = endpoint.send(endpoint.request("/search/",
                                                                          Map.of("yql", yql,
                                                                                 "timeout", "10s")));
        assertEquals(200, emptyResult.statusCode());
        assertEquals(0, new ObjectMapper().readTree(emptyResult.body())
                .get("root").get("fields").get("totalCount").asLong());

        HttpResponse<String> feedResult = endpoint.send(endpoint.request(documentPath)
                                               .POST(ofString(document)));
        assertEquals(200, feedResult.statusCode());

        HttpResponse<String> searchResult = endpoint.send(endpoint.request("/search/",
                                                                           Map.of("yql", yql)));
        assertEquals(200, searchResult.statusCode());
        assertEquals(1, new ObjectMapper().readTree(searchResult.body())
                .get("root").get("fields").get("totalCount").asLong());
    }
}
