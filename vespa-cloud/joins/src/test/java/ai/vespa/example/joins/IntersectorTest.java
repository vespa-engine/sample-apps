// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.joins;

import ai.vespa.example.joins.Intersector.Intersection;
import ai.vespa.example.joins.Intersector.Interval;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class IntersectorTest {

    @Test
    void testIntersection() {
        List<MyInterval> first = List.of(new MyInterval("a", 0, 3),
                                         new MyInterval("b", 4, 7),
                                         new MyInterval("c", 10, 12),
                                         new MyInterval("d", 15, 18),
                                         new MyInterval("e", 20, 25));

        List<MyInterval> second = List.of(new MyInterval("A", 0, 3),
                                          new MyInterval("B", 5, 5),
                                          new MyInterval("C", 8, 10),
                                          new MyInterval("DE", 17, 21),
                                          new MyInterval("X", 26, 27));

        List<Intersection<MyInterval, MyInterval>> intersections = Intersector.intersect(first, second);
        Intersection<MyInterval, MyInterval> intersection;

        intersection = intersections.remove(0);
        assertEquals(0, intersection.start());
        assertEquals(3, intersection.end());
        assertEquals(first.get(0), intersection.first());
        assertEquals(second.get(0), intersection.second());

        intersection = intersections.remove(0);
        assertEquals(5, intersection.start());
        assertEquals(5, intersection.end());
        assertEquals(first.get(1), intersection.first());
        assertEquals(second.get(1), intersection.second());

        intersection = intersections.remove(0);
        assertEquals(10, intersection.start());
        assertEquals(10, intersection.end());
        assertEquals(first.get(2), intersection.first());
        assertEquals(second.get(2), intersection.second());

        intersection = intersections.remove(0);
        assertEquals(17, intersection.start());
        assertEquals(18, intersection.end());
        assertEquals(first.get(3), intersection.first());
        assertEquals(second.get(3), intersection.second());

        intersection = intersections.remove(0);
        assertEquals(20, intersection.start());
        assertEquals(21, intersection.end());
        assertEquals(first.get(4), intersection.first());
        assertEquals(second.get(3), intersection.second());

        assertEquals(List.of(), intersections);
    }

    static class MyInterval implements Interval {

        final String id;
        final long start;
        final long end;

        public MyInterval(String id, long start, long end) {
            this.id = id;
            this.start = start;
            this.end = end;
        }

        @Override
        public long start() {
            return start;
        }

        @Override
        public long end() {
            return end;
        }

        @Override
        public String toString() {
            return "MyInterval{" +
                   "id='" + id + '\'' +
                   ", start=" + start +
                   ", end=" + end +
                   '}';
        }

    }

}
