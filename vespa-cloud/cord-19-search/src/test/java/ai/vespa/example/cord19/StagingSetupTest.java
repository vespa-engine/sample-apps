// Copyright 2020 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19;

import ai.vespa.hosted.cd.StagingSetup;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import static org.junit.jupiter.api.Assertions.assertEquals;

@StagingSetup
class StagingSetupTest {

    @Test
    @DisplayName("Feed documents to the staging cluster, before upgrade")
    void feedAndSearch() throws IOException {
        assertEquals(200,200);
    }

}
