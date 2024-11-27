// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

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
import static ai.vespa.example.album.StagingCommons.documentsByPath;
import static ai.vespa.example.album.StagingCommons.verifyDocumentsAreSearchable;
import static org.junit.jupiter.api.Assertions.assertEquals;

@StagingTest
class StagingVerificationTest {

    @Test
    @DisplayName("Verify documents can be searched after upgrade")
    void verify() throws IOException {
        // Verify each of the documents fed in setup are still there, with expected data.
        ObjectMapper mapper = new ObjectMapper();
        documentsByPath().forEach((documentPath, document) -> {
            try {
                HttpResponse<String> documentResponse = container().send(container().request(documentPath).GET());
                assertEquals(200, documentResponse.statusCode());
                JsonNode retrieved = mapper.readTree(documentResponse.body()).get("fields");
                JsonNode expected = mapper.readTree(document).get("fields");
                for (String name : List.of("text"))
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
