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
        String documentPath = "/document/v1/joins/title/docid/test1";
        String document = "{\n" +
                          "    \"fields\": {\n" +
                          "         \"text\": \"Sample text\"\n" +
                          "    }\n" +
                          "}\n";

        String yql = "SELECT * FROM SOURCES * WHERE title CONTAINS 'text'";

        HttpResponse<String> deleteResult = endpoint.send(endpoint.request(documentPath).DELETE());
        assertEquals(200, deleteResult.statusCode());

        // the first query needs a higher timeout than the default 500ms, to warm up the code
        HttpResponse<String> emptyResult = endpoint.send(endpoint.request("/search/",
                                                                          Map.of("yql", yql,
                                                                                 "timeout", "5s")));
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

    // Add more tests, for inspiration use
    // https://github.com/vespa-engine/sample-apps/tree/3a05197d549bf36990651636573a1bd810b86c4c/basic-search-hosted and
    // https://github.com/vespa-engine/sample-apps/tree/e691b706d84b73caa3f5a595ae6c91388a4c2ca0/basic-search-java

}
