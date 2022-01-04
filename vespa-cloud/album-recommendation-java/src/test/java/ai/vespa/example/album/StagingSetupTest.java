// Copyright 2020 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import ai.vespa.feed.client.DocumentId;
import ai.vespa.feed.client.FeedClient;
import ai.vespa.feed.client.FeedClientBuilder;
import ai.vespa.feed.client.JsonFeeder;
import ai.vespa.feed.client.OperationParameters;
import ai.vespa.feed.client.Result;
import ai.vespa.hosted.cd.StagingSetup;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import static ai.vespa.example.album.StagingCommons.container;
import static ai.vespa.example.album.StagingCommons.documents;
import static ai.vespa.example.album.StagingCommons.verifyDocumentsAreSearchable;

@StagingSetup
class StagingSetupTest {

    @Test
    @DisplayName("Feed documents to the staging cluster, before upgrade")
    void feedAndSearch() throws IOException, ExecutionException, InterruptedException {
        // Feed the static staging test documents; staging clusters are always empty when setup is run.
        FeedClient feedClient = FeedClientBuilder.create(container().uri()).build();
        JsonFeeder jsonFeeder = JsonFeeder.builder(feedClient).build();
        for (String document : documents()) {
            jsonFeeder.feedSingle(document).get();
        }

        // Verify documents are searchable and rendered as expected, prior to upgrade.
        verifyDocumentsAreSearchable();
    }

}
