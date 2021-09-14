// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.joins;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Intersector {

    private Intersector() { }

    /**
     * Iterates through the given intervals, returning the set of intersections between them.
     * The given intervals must be non-overlapping within each iterator, and sorted by increasing time.
     */
    public static <S extends Interval, T extends Interval> List<Intersection<S, T>> intersect(Iterable<S> first, Iterable<T> second) {
        Iterator<S> sit = first.iterator();
        Iterator<T> tit = second.iterator();
        List<Intersection<S, T>> intersections = new ArrayList<>();
        if ( ! sit.hasNext() || ! tit.hasNext())
            return intersections;

        S s = sit.next();
        T t = tit.next();
        while (true) {
            long start = Math.max(s.start(), t.start());
            long end = Math.min(s.end(), t.end());
            if (start <= end)
                intersections.add(new Intersection<>(start, end, s, t));

            if (s.end() < t.end()) {
                if (sit.hasNext()) s = sit.next();
                else break;
            }
            else {
                if (tit.hasNext()) t = tit.next();
                else break;
            }
        }
        return intersections;
    }


    public interface Interval {

        long start();

        long end();

    }

    public static class Intersection<S extends Interval, T extends Interval> implements Interval {

        private final long start;
        private final long end;
        private final S first;
        private final T second;

        public Intersection(long start, long end, S first, T second) {
            this.start = start;
            this.end = end;
            this.first = first;
            this.second = second;
        }

        @Override
        public long start() {
            return start;
        }

        @Override
        public long end() {
            return end;
        }

        public S first() { return first; }

        public T second() { return second; }

        @Override
        public String toString() {
            return "Intersection{" +
                   "start=" + start +
                   ", end=" + end +
                   ", first=" + first +
                   ", second=" + second +
                   '}';
        }

    }

}
