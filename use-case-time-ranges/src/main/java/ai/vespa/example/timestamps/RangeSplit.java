// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.timestamps;

import java.util.*;

/**
 * @author arnej
 */
public class RangeSplit {

    // Check from coarsest (fewest bits, largest range) to finest
    static private final int[] bitLevels = {1, 9, 17, 25, 33, 38, 43, 48, 52, 56, 60};

    static class MarkBuilder {
        private final StringBuilder buf = new StringBuilder();
        MarkBuilder bits(int bits) {
            buf.append("B");
            buf.append(String.format("%02d", bits));
            return this;
        }
        MarkBuilder limit(long limit) {
            buf.append("L");
            // Flip the sign bit to map Long.MIN_VALUE to 0x0000000000000000
            // and Long.MAX_VALUE to 0xffffffffffffffff
            long adjusted = limit ^ Long.MIN_VALUE;
            buf.append(String.format("%016x", adjusted));
            return this;
        }
        String build() { return buf.toString(); }
    }

    /**
     * Calculate the range bucket floor for a given value at specified bit precision.
     * @param value the long value to bucket
     * @param topBits number of top bits to preserve (1-64)
     * @return the floor value of the range bucket
     */
    private static long getRangeBucketFloor(long value, int topBits) {
        if (topBits <= 0 || topBits > 64) {
            throw new IllegalArgumentException("topBits must be between 1 and 64");
        }
        if (topBits == 64) {
            return value; // All bits preserved
        }

        // Zero out the lower bits by shifting right then left
        int shiftAmount = 64 - topBits;
        return (value >> shiftAmount) << shiftAmount;
    }

    /**
     * Generate range markers for a given number.
     * Creates markers at bit precision levels: 1, 9, 17, 25, 33, 38, 43, 48, 52, 56, 60.
     * @param value the long value to generate markers for
     * @return list of range marker strings
     */
    public static List<String> generateRangeMarks(long value) {
        List<String> marks = new ArrayList<>(bitLevels.length);

        for (int bits : bitLevels) {
            long bucketFloor = getRangeBucketFloor(value, bits);
            String mark = new MarkBuilder().bits(bits).limit(bucketFloor).build();
            marks.add(mark);
        }

        return marks;
    }

    /**
     * Round a value down so the lowest 4 bits are zero.
     * @param value the value to round down
     * @return the rounded value
     */
    private static long roundDownTo4Bits(long value) {
        return (value >> 4) << 4;
    }

    /**
     * Round a value up so the lowest 4 bits are all ones.
     * @param value the value to round up
     * @return the rounded value
     */
    private static long roundUpTo4Bits(long value) {
        return value | 0xF;
    }

    /**
     * Find the extra range at the start that was included due to rounding down.
     * Returns [roundedLo, lo) as a range that should be excluded.
     * @param lo the original lower bound
     * @return the extra range at the start, or null if there is no extra range
     */
    public static ExtraRange findExtraRangeAtStart(long lo) {
        long roundedLo = roundDownTo4Bits(lo);
        if (roundedLo < lo) {
            return new ExtraRange(roundedLo, lo - 1);
        }
        return null;
    }

    /**
     * Find the extra range at the end that was included due to rounding up.
     * Returns (hi, roundedHi] as a range that should be excluded.
     * @param hi the original upper bound
     * @return the extra range at the end, or null if there is no extra range
     */
    public static ExtraRange findExtraRangeAtEnd(long hi) {
        long roundedHi = roundUpTo4Bits(hi);
        if (hi < roundedHi) {
            return new ExtraRange(hi + 1, roundedHi);
        }
        return null;
    }

    /**
     * Represents a range [lo, hi] that was added due to rounding and should be excluded.
     */
    public static class ExtraRange {
        public final long lo;
        public final long hi;

        public ExtraRange(long lo, long hi) {
            this.lo = lo;
            this.hi = hi;
        }

        @Override
        public String toString() {
            return "[" + lo + ", " + hi + "]";
        }
    }

    /**
     * Find the smallest number of bits (largest range) such that the range bucket starting at 'start'
     * with that precision does not exceed 'limit'.
     * @param start the starting position
     * @param limit the maximum ending position (inclusive)
     * @return the number of bits for the largest fitting range, or -1 if none fit
     */
    private static int findLargestFittingRange(long start, long limit) {
        for (int bits : bitLevels) {
            long bucketFloor = getRangeBucketFloor(start, bits);
            if (bucketFloor != start) {
                // start is not aligned to this bit level
                continue;
            }

            // Calculate the end of this range bucket
            int shiftAmount = 64 - bits;
            long rangeSize = 1L << shiftAmount;
            long bucketEnd = start + rangeSize - 1;

            if (bucketEnd <= limit) {
                return bits;
            }
        }

        return -1; // No fitting range found
    }

    /**
     * Generate a set of range markers that cover the inclusive range [lo, hi].
     * First rounds lo down and hi up to 4-bit boundaries, then finds the minimal
     * set of range markers that cover the entire range.
     * @param lo the lower bound (inclusive)
     * @param hi the upper bound (inclusive)
     * @return list of range marker strings covering the range
     */
    public static List<String> generateCoveringRangeMarks(long lo, long hi) {
        if (lo > hi) {
            throw new IllegalArgumentException("lo must be <= hi");
        }

        // Round to 4-bit boundaries
        long roundedLo = roundDownTo4Bits(lo);
        long roundedHi = roundUpTo4Bits(hi);

        List<String> marks = new ArrayList<>();
        long current = roundedLo;

        while (current <= roundedHi) {
            int bits = findLargestFittingRange(current, roundedHi);

            if (bits == -1) {
                // No aligned range found at current position, use finest granularity (60 bits)
                bits = 60;
                long bucketFloor = getRangeBucketFloor(current, bits);
                String mark = new MarkBuilder().bits(bits).limit(bucketFloor).build();
                marks.add(mark);

                // Move forward by the 60-bit range size (16 values, since 64-60=4 bits)
                current += 16;
                if (current > roundedHi) {
                    break;
                }
                continue;
            }

            // Add the range marker
            String mark = new MarkBuilder().bits(bits).limit(current).build();
            marks.add(mark);

            // Move to the next position
            int shiftAmount = 64 - bits;
            long rangeSize = 1L << shiftAmount;
            long nextCurrent = current + rangeSize;

            // Check for overflow or if we've covered everything
            if (nextCurrent <= current || nextCurrent > roundedHi + 1) {
                break;
            }

            current = nextCurrent;
        }

        return marks;
    }


}
