// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.mydomain.example;

import com.yahoo.application.Application;
import com.yahoo.application.Networking;
import com.yahoo.application.container.Search;
import com.yahoo.component.ComponentSpecification;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import org.junit.Test;

import java.nio.file.FileSystems;

import static org.junit.Assert.assertEquals;

/**
 * Unit test of the container aspect of this application.
 */
public class ApplicationTest {

    @Test
    public void testApplication() {
        try (Application app = Application.fromApplicationPackage(
                FileSystems.getDefault().getPath("src/main/application"),
                Networking.disable)) {
            Search search = app.getJDisc("jdisc").search();
            Result result = search.process(ComponentSpecification.fromString("default"), new Query());
            assertEquals("Artificial hit is added",
                         "test:hit", result.hits().get(0).getId().toString());
        }
    }

}
