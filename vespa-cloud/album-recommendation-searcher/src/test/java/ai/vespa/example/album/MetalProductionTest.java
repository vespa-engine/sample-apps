package ai.vespa.example.album;

import ai.vespa.hosted.cd.ProductionTest;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ProductionTest
public class MetalProductionTest {

    @Test
    void dummyTest() {
        assertEquals("Prod is all right!","Prod is all right!");
    }
}
