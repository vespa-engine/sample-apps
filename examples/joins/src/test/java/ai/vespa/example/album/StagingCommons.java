// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.TestRuntime;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.http.HttpResponse;
import java.util.Map;
import java.util.stream.Stream;

import static java.net.URLEncoder.encode;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.toUnmodifiableMap;
import static org.junit.jupiter.api.Assertions.assertEquals;

class StagingCommons {

    private static final ObjectMapper mapper = new ObjectMapper();

    /** Returns the container endpoint to do requests against. */
    static Endpoint container() {
        return TestRuntime.get().deploymentToTest().endpoint("default");
    }

    /** Returns the document path of the document with the given name. */
    static String documentPath(String documentName) {
        return "/document/v1/staging/title/docid/" + encode(documentName, UTF_8);
    }

    /** Reads and returns the contents of the JSON test resource with the given name. */
    static byte[] readDocumentResource(String documentName) {
        try {
            return StagingSetupTest.class.getResourceAsStream("/" + documentName + ".json").readAllBytes();
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Returns static document ID paths and document bytes for the three static staging test documents. */
    static Map<String, byte[]> documentsByPath() {
        return Stream.of("Title1",
                         "Title2",
                         "Title3")
                     .collect(toUnmodifiableMap(StagingCommons::documentPath,
                                                StagingCommons::readDocumentResource));
    }

    /** Warm-up query matching all "music" documents — high timeout as the fresh container needs to warm up. */
    static Map<String, String> warmupQueryForAllDocuments() {
        return Map.of("yql", "SELECT * FROM SOURCES * WHERE sddocname CONTAINS 'title'", "timeout", "5s");
    }

    static Map<String, String> queryForTitles() {
        return Map.of("yql", "SELECT * FROM SOURCES * WHERE text contains 'generic'");
    }

    /** Verifies the static staging documents are searchable, ranked correctly, and render as expected. */
    static void verifyDocumentsAreSearchable() throws IOException {
        warmup();

        // Verify that the cluster filters and ranks documents as expected, prior to upgrade.
        HttpResponse<String> queryResponse = container().send(container().request("/search/", queryForTitles()));
        assertEquals(200, queryResponse.statusCode());
        JsonNode root = mapper.readTree(queryResponse.body()).get("root");
        assertEquals(2, root.get("fields").get("totalCount").asLong());
    }

    private static void warmup() throws IOException {
        // Verify that the cluster has the fed documents, and that they are searchable.
        for (int i = 0; i <= 5; i++) {
            HttpResponse<String> warmUpResponse = container().send(container().request("/search/", warmupQueryForAllDocuments()));
            assertEquals(200, warmUpResponse.statusCode());
            assertEquals(3, mapper.readTree(warmUpResponse.body())
                    .get("root").get("fields").get("totalCount").asLong());
        }
    }

}
