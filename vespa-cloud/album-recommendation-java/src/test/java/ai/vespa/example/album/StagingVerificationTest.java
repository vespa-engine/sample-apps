// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import ai.vespa.feed.client.DocumentId;
import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.StagingTest;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.http.HttpResponse;
import java.util.Iterator;
import java.util.List;

import static ai.vespa.example.album.StagingCommons.container;
import static ai.vespa.example.album.StagingCommons.documents;
import static ai.vespa.example.album.StagingCommons.verifyDocumentsAreSearchable;
import static org.junit.jupiter.api.Assertions.assertEquals;

@StagingTest
class StagingVerificationTest {

    @Test
    @DisplayName("Verify documents can be searched after upgrade")
    void verify() throws IOException {
        // Verify each of the documents fed in setup are still there, with expected data.
        ObjectMapper mapper = new ObjectMapper();
        Endpoint container = container();
        documents().forEach(document -> {
            try {
                JsonNode json = mapper.readTree(document);
                DocumentId documentId = DocumentId.of(json.get("put").asText());
                String path = StagingCommons.documentPath(documentId);
                HttpResponse<String> documentResponse = container.send(container.request(path).GET());
                assertEquals(200, documentResponse.statusCode());
                JsonNode retrieved = mapper.readTree(documentResponse.body()).get("fields");
                JsonNode expected = json.get("fields");
                for (String name : List.of("artist", "album", "year"))
                    assertEquals(expected.get(name), retrieved.get(name));
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });

        // Verify documents are searchable and rendered as expected, after the upgrade.
        verifyDocumentsAreSearchable();
    }

}
