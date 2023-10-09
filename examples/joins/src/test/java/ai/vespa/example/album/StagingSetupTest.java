// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import ai.vespa.hosted.cd.StagingSetup;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static ai.vespa.example.album.StagingCommons.container;
import static ai.vespa.example.album.StagingCommons.documentsByPath;
import static ai.vespa.example.album.StagingCommons.verifyDocumentsAreSearchable;
import static java.net.http.HttpRequest.BodyPublishers.ofByteArray;
import static org.junit.jupiter.api.Assertions.assertEquals;

@StagingSetup
class StagingSetupTest {

    @Test
    @DisplayName("Feed documents to the staging cluster, before upgrade")
    void feedAndSearch() throws IOException {
        // Feed the static staging test documents; staging clusters are always empty when setup is run.
        documentsByPath().forEach((documentPath, document) -> {
            assertEquals(200, container().send(container().request(documentPath).POST(ofByteArray(document))).statusCode());
        });

        // Verify documents are searchable and rendered as expected, prior to upgrade.
        verifyDocumentsAreSearchable();
    }

}
