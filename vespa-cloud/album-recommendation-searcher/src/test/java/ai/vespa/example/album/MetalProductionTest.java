// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import ai.vespa.hosted.cd.ProductionTest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ProductionTest
public class MetalProductionTest {

    @Test
    void dummyTest() {
        assertEquals("Prod is all right!", "Prod is all right!");
    }

}
