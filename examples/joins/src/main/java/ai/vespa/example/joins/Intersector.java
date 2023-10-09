// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.joins;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;

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


    /**
     * Iterates through the given intervals, returning the set of intersections between them.
     * The given intervals may be overlapping within each iterator, but must be sorted by increasing start time.
     */
    public static <S extends Interval, T extends Interval> List<Intersection<S, T>> intersectX(Iterable<S> first, Iterable<T> second) {
        Iterator<S> sit = first.iterator();
        Iterator<T> tit = second.iterator();
        List<Intersection<S, T>> intersections = new ArrayList<>();
        if ( ! sit.hasNext() || ! tit.hasNext())
            return intersections;

        PriorityQueue<S> ses = new PriorityQueue<>(Comparator.comparingLong(Interval::end));
        PriorityQueue<T> tes = new PriorityQueue<>(Comparator.comparingLong(Interval::end));

        S s = sit.next();
        T t = tit.next();
        while (s != null || t != null) {
            // Figure out what to do: add a new interval, or close an open one; and in any case, from which set?
            long end = Math.min(ses.isEmpty() ? Long.MAX_VALUE : ses.peek().end(),
                                tes.isEmpty() ? Long.MAX_VALUE : tes.peek().end());
            long start = Math.min(s == null ? Long.MAX_VALUE : s.start(),
                                  t == null ? Long.MAX_VALUE : t.start());

            // Remove any done interval from the sets of open intervals ...
            if (end < start) {
                if ( ! ses.isEmpty() && ses.peek().end() == end) ses.poll();
                if ( ! tes.isEmpty() && tes.peek().end() == end) tes.poll();
            }
            // ... or add a new interval to these sets, and add intersection with open intervals from the other iterator.
            else {
                if (s != null && s.start() == start) {
                    ses.add(s);
                    for (T ot : tes) intersections.add(new Intersection<>(start, Math.min(s.end(), ot.end()), s, ot));
                    s = sit.hasNext() ? sit.next() : null;
                }
                else if (t != null) {
                    tes.add(t);
                    for (S os : ses) intersections.add(new Intersection<>(start, Math.min(os.end(), t.end()), os, t));
                    t = tit.hasNext() ? tit.next() : null;
                }
                else throw new IllegalStateException("Should not happen");
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
            if (start > end)
                throw new IllegalArgumentException("Non-intersecting intervals: " + first + ", " + second);

            this.start = start;
            this.end = end;
            this.first = first;
            this.second = second;
        }

        public Intersection(S first, T second) {
            this(Math.max(first.start(), second.start()), Math.min(first.end(), second.end()), first, second);
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
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Intersection<?, ?> that = (Intersection<?, ?>) o;
            return start == that.start && end == that.end && first.equals(that.first) && second.equals(that.second);
        }

        @Override
        public int hashCode() {
            return Objects.hash(start, end, first, second);
        }

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
