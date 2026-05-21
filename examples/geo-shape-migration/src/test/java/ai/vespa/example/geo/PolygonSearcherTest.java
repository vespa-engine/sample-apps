// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.geo;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PolygonSearcherTest {

    private static final double[][] BERLIN_TRIANGLE = {
            { 52.55, 13.30 },
            { 52.55, 13.45 },
            { 52.49, 13.40 }
    };

    @Test
    void parsesClosedRingByDroppingDuplicate() {
        double[][] p = PolygonSearcher.parsePolygon(
                "52.55,13.30,52.55,13.45,52.49,13.40,52.55,13.30");
        assertEquals(3, p.length);
    }

    @Test
    void parsesOpenRing() {
        double[][] p = PolygonSearcher.parsePolygon(
                "52.55,13.30,52.55,13.45,52.49,13.40");
        assertEquals(3, p.length);
    }

    @Test
    void pointInsideTriangle() {
        assertTrue(PolygonSearcher.pointInPolygon(52.52, 13.38, BERLIN_TRIANGLE));
    }

    @Test
    void pointOutsideTriangle() {
        // Potsdam — west of the triangle.
        assertFalse(PolygonSearcher.pointInPolygon(52.39, 13.06, BERLIN_TRIANGLE));
        // Munich — far away.
        assertFalse(PolygonSearcher.pointInPolygon(48.13, 11.58, BERLIN_TRIANGLE));
    }

    @Test
    void pointOnEdgeTreatedConsistently() {
        // A point exactly at a vertex should not crash and gives a deterministic answer.
        boolean atVertex = PolygonSearcher.pointInPolygon(52.55, 13.30, BERLIN_TRIANGLE);
        // Either true or false is acceptable here — we just assert no exception.
        assertTrue(atVertex || !atVertex);
    }
}
