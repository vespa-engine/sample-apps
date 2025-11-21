// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.timestamps;

import com.yahoo.prelude.query.Limit;
import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.IntItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.prelude.query.NotItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.PhaseNames;
import com.yahoo.prelude.query.QueryCanonicalizer;
import com.yahoo.yolean.chain.After;
import com.yahoo.yolean.chain.Before;

import ai.vespa.example.timestamps.RangesplitterConfig;
import java.util.*;

/**
 * Splits a range into a simple OR
 * @author arnej
 */
@After(PhaseNames.TRANSFORMED_QUERY)
public class RangeSplitterSearcher extends Searcher {

    /**
     * Represents a field pair configuration for range splitting.
     * @param originalField the field where ranges can be split
     * @param rangeMarkerField the corresponding field where range markers are indexed
     */
    public record FieldPair(String originalField, String rangeMarkerField) {
        @Override
        public String toString() {
            return originalField + " -> " + rangeMarkerField;
        }
    }

    private final List<FieldPair> fieldPairs;

    /**
     * Construct a RangeSplitter from configuration.
     * @param config the configuration specifying which fields to split
     */
    public RangeSplitterSearcher(RangesplitterConfig config) {
        this.fieldPairs = new ArrayList<>();
        for (RangesplitterConfig.Splitfields splitfield : config.splitfields()) {
            fieldPairs.add(new FieldPair(splitfield.originalField(), splitfield.rangeMarkerField()));
        }
    }

    /**
     * Get the configured field pairs.
     * @return list of field pairs
     */
    public List<FieldPair> getFieldPairs() {
        return Collections.unmodifiableList(fieldPairs);
    }

    @Override
    public Result search(Query query, Execution execution) {
// System.err.println("running with " + fieldPairs.size() + " field pairs");
        if (fieldPairs.isEmpty()) {
            return execution.search(query);
        }

        Item root = query.getModel().getQueryTree().getRoot();
        if (root != null) {
            // Handle the case where root itself is an IntItem
// System.out.println("process root: " + root.getClass());
            if (root instanceof IntItem intItem) {
                Item replacement = processIntItem(intItem, query);
                if (replacement != null) {
                    query.getModel().getQueryTree().setRoot(replacement);
                }
            } else {
                findAndProcessRangeItems(root, query);
            }
        }

        return execution.search(query);
    }

    /**
     * Recursively find all IntItem ranges in configured fields and replace them with OR of markers.
     */
    private void findAndProcessRangeItems(Item item, Query query) {
        if (item instanceof CompositeItem compositeItem) {
// System.out.println("process composite: " + item.getClass());
            List<Item> replacements = new ArrayList<>();

            // First pass: identify and remove items to be replaced
            for (Iterator<Item> i = compositeItem.getItemIterator(); i.hasNext(); ) {
                Item child = i.next();
                i.remove();
                Item replacement = null;
// System.out.println("check child: " + child.getClass());

                if (child instanceof IntItem intItem) {
                    replacement = processIntItem(intItem, query);
                } else {
                    // Recurse into nested composite items
                    findAndProcessRangeItems(child, query);
                }
                if (replacement != null) {
                    replacements.add(replacement);
                } else {
                    replacements.add(child);
                }
            }

            // Second pass: add all replacement items
            for (Item replacement : replacements) {
                compositeItem.addItem(replacement);
            }
        }
    }

    /**
     * Process an IntItem if it's a range query on a configured field.
     * Returns a replacement item (OrItem with range markers) or null if no replacement needed.
     */
    private Item processIntItem(IntItem item, Query query) {
        String fieldName = item.getIndexName();

        // Check if this field is configured for range splitting
        FieldPair matchingPair = findFieldPair(fieldName);
// System.out.println("findFieldPair("+fieldName+" -> " + matchingPair);
        if (matchingPair == null) {
            return null;
        }

        // Check if it's actually a range (not a single value)
        if (item.getFromLimit().equals(item.getToLimit())) {
            return null;
        }

        // Print details about this range item
        System.out.println("Found range item:");
        System.out.println("  Field: " + fieldName + " (will use marker field: " + matchingPair.rangeMarkerField() + ")");
        System.out.println("  Range: [" + item.getFromLimit() + ", " + item.getToLimit() + "]");
        System.out.println("  From limit: " + item.getFromLimit().number() + " (inclusive=" + item.getFromLimit().isInclusive() + ")");
        System.out.println("  To limit: " + item.getToLimit().number() + " (inclusive=" + item.getToLimit().isInclusive() + ")");

        // Convert limits to long values
        long fromValue = item.getFromLimit().number().longValue();
        long toValue = item.getToLimit().number().longValue();

        // Adjust for inclusiveness
        if (!item.getFromLimit().isInclusive()) {
            fromValue++;
        }
        if (!item.getToLimit().isInclusive()) {
            toValue--;
        }

        // Generate covering range marks
        List<String> markers = RangeSplit.generateCoveringRangeMarks(fromValue, toValue);
        System.out.println("  Generated " + markers.size() + " range markers");

        // Check for extra ranges due to rounding
        RangeSplit.ExtraRange extraStart = RangeSplit.findExtraRangeAtStart(fromValue);
        RangeSplit.ExtraRange extraEnd = RangeSplit.findExtraRangeAtEnd(toValue);

        if (extraStart != null) {
            System.out.println("  Extra range at start: " + extraStart);
        }
        if (extraEnd != null) {
            System.out.println("  Extra range at end: " + extraEnd);
        }

        // Create OR item with all markers
        OrItem orItem = new OrItem();
        for (String marker : markers) {
            WordItem markerItem = new WordItem(marker, matchingPair.rangeMarkerField());
            markerItem.setRanked(false);
            orItem.addItem(markerItem);
        }

        orItem.setRanked(false);

        // If we have extra ranges, wrap in NOT to exclude them
        if (extraStart != null || extraEnd != null) {
            NotItem notItem = new NotItem();
            notItem.addItem(orItem);  // First item is the positive item

            // Add extra ranges as negative IntItem instances on the original field
            if (extraStart != null) {
                IntItem excludeItem = IntItem.from(fieldName,
                    new Limit(extraStart.lo, true),
                    new Limit(extraStart.hi, true),
                    0);
                excludeItem.setRanked(false);
                notItem.addNegativeItem(excludeItem);
                System.out.println("    Excluding start extra range: " + fieldName + ":[" + extraStart.lo + ".." + extraStart.hi + "]");
            }

            if (extraEnd != null) {
                IntItem excludeItem = IntItem.from(fieldName,
                    new Limit(extraEnd.lo, true),
                    new Limit(extraEnd.hi, true),
                    0);
                excludeItem.setRanked(false);
                notItem.addNegativeItem(excludeItem);
                System.out.println("    Excluding end extra range: " + fieldName + ":[" + extraEnd.lo + ".." + extraEnd.hi + "]");
            }

            notItem.setRanked(false);
            System.out.println("  Replaced with NOT wrapping OR of " + markers.size() + " markers on field " + matchingPair.rangeMarkerField());
            System.out.println();
            return notItem;
        }

        System.out.println("  Replaced with OR of " + markers.size() + " markers on field " + matchingPair.rangeMarkerField());
        System.out.println();

        return orItem;
    }

    /**
     * Find a configured field pair for the given field name.
     */
    private FieldPair findFieldPair(String fieldName) {
        for (FieldPair pair : fieldPairs) {
            if (pair.originalField().equals(fieldName)) {
                return pair;
            }
        }
        return null;
    }

}
