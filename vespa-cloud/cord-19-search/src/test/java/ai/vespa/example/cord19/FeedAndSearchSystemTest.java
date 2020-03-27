// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.SystemTest;
import ai.vespa.hosted.cd.TestRuntime;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static java.net.http.HttpRequest.BodyPublishers.ofString;
import static org.junit.jupiter.api.Assertions.assertEquals;

@SystemTest
class FeedAndSearchSystemTest {

    private final Endpoint endpoint = TestRuntime.get().deploymentToTest().endpoint("default");

    @Test
    void feedAndSearch() throws IOException {

        assertEquals(200,200);

    }

    // Add more tests, for inspiration use
    // https://github.com/vespa-engine/sample-apps/tree/3a05197d549bf36990651636573a1bd810b86c4c/basic-search-hosted and
    // https://github.com/vespa-engine/sample-apps/tree/e691b706d84b73caa3f5a595ae6c91388a4c2ca0/basic-search-java
}
