// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.SystemTest;
import ai.vespa.hosted.cd.StagingTest;
import ai.vespa.hosted.cd.TestRuntime;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.http.HttpResponse;
import java.util.Map;

import static java.net.http.HttpRequest.BodyPublishers.ofString;
import static org.junit.jupiter.api.Assertions.assertEquals;

@SystemTest
@StagingTest
class FeedAndSearchSystemTest {

    private final Endpoint endpoint = TestRuntime.get().deploymentToTest().endpoint("default");

    @Test
    void feedAndSearch() throws IOException {
        String documentPath = "/document/v1/mynamespace/music/docid/Got-to-be-there";
        String document = "{\n"
                     + "    \"fields\": {\n"
                     + "         \"album\": \"Got to be there\"\n"
                     + "    }\n"
                     + "}\n";

        String yql = "SELECT * FROM SOURCES * WHERE album CONTAINS \"Got to be there\";";

        HttpResponse<String> deleteResult = endpoint.send(endpoint.request(documentPath).DELETE());
        assertEquals(200, deleteResult.statusCode());

        // the first query needs a higher timeout than the default 500ms, to warm up the code
        HttpResponse<String> emptyResult = endpoint.send(endpoint.request("/search/", Map.of("yql", yql,
                                                                            "timeout", "5s")));
        assertEquals(200, emptyResult.statusCode());
        assertEquals(0, new ObjectMapper().readTree(emptyResult.body())
                .get("root").get("fields").get("totalCount").asLong());

        HttpResponse<String> feedResult = endpoint.send(endpoint.request(documentPath)
                                               .POST(ofString(document)));
        assertEquals(200, feedResult.statusCode());

        HttpResponse<String> searchResult = endpoint.send(endpoint.request("/search/", Map.of("yql", yql)));
        assertEquals(200, searchResult.statusCode());
        assertEquals(1, new ObjectMapper().readTree(searchResult.body())
                .get("root").get("fields").get("totalCount").asLong());
    }

}
