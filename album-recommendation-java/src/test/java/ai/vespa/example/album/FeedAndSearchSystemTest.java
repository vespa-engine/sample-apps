// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.SystemTest;
import ai.vespa.hosted.cd.StagingTest;
import ai.vespa.hosted.cd.TestRuntime;
import com.yahoo.slime.JsonDecoder;
import com.yahoo.slime.Slime;
import org.junit.jupiter.api.Test;

import java.net.http.HttpResponse;
import java.util.Map;

import static java.net.http.HttpRequest.BodyPublishers.ofString;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SystemTest
@StagingTest
class FeedAndSearchSystemTest {

    private final Endpoint endpoint = TestRuntime.get().deploymentToTest().endpoint("default");

    @Test
    void feedAndSearch() {
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
        
        var emptyInspector = new JsonDecoder().decode(new Slime(), emptyResult.body().getBytes(UTF_8)).get();
        assertEquals(200, emptyResult.statusCode());
        assertTrue(emptyInspector.field("root").field("fields").field("totalCount").valid());
        assertEquals(0, emptyInspector.field("root").field("fields").field("totalCount").asLong());

        HttpResponse<String> feedResult = endpoint.send(endpoint.request(documentPath)
                                               .POST(ofString(document)));
        assertEquals(200, feedResult.statusCode());

        HttpResponse<String> searchResult = endpoint.send(endpoint.request("/search/", Map.of("yql", yql)));
        var searchInspector = new JsonDecoder().decode(new Slime(), searchResult.body().getBytes(UTF_8)).get();
        assertEquals(200, searchResult.statusCode());
        assertTrue(searchInspector.field("root").field("fields").field("totalCount").valid());
        assertEquals(1, searchInspector.field("root").field("fields").field("totalCount").asLong());
    }

}
