// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.joins;

import ai.vespa.example.joins.Intersector.Intersection;
import ai.vespa.example.joins.Intersector.Interval;
import com.yahoo.prelude.query.AndItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.Limit;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.prelude.query.RangeItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class JoinSearcher extends Searcher {

    private static final String ATTRIBUTE_ONLY_SUMMARY_CLASS = "attribute-only";

    @Override
    public Result search(Query query, Execution execution) {
        Optional<JoinSpec> spec = JoinSpec.from(query);
        if (spec.isEmpty()) return execution.search(query); // No join spec so skip this searcher

        Result aResult = execution.search(spec.get().addATerms(query.clone()));
        execution.fill(aResult, ATTRIBUTE_ONLY_SUMMARY_CLASS);

        switch (spec.get().variant) {
            case queryPerHit: return joinWithQueryPerHit(query.clone(), aResult, spec.get(), execution);
            case rangeItemPerHit: return joinWithRangeItemPerHit(query.clone(), aResult, spec.get(), execution);
            case containerIntersect: return joinInContainer(query.clone(), aResult, spec.get(), execution);
            default: throw new IllegalStateException("Unexpected join spec variant: " + spec.get().variant);
        }
    }

    private Result joinWithQueryPerHit(Query query, Result aResult, JoinSpec spec, Execution execution) {
        Result result = new Result(query);
        for (Hit hit : aResult.hits()) {
            if (hit.isAuxiliary()) {
                result.hits().add(hit);
            }
            else {
                for (Hit joinedHit : joinWithQuery(hit, query.clone(), spec, execution))
                    result.hits().add(joinedHit);
            }
        }
        return result;
    }

    private List<Hit> joinWithQuery(Hit aHit, Query query, JoinSpec spec, Execution execution) {
        spec.addBTerms(query);
        query.getModel().getQueryTree().and(new WordItem(aHit.getField("id").toString(), "id"));
        spec.addInterval((long)aHit.getField("start"), (long)aHit.getField("end"), query);
        Result bResult = execution.search(query);
        execution.fill(bResult, ATTRIBUTE_ONLY_SUMMARY_CLASS);
        List<Hit> joinedHits = new ArrayList<>();
        for (Hit bHit : bResult.hits()) {
            Hit joinedHit = aHit.clone();
            joinedHit.setField("tagid", bHit.getField("tagid"));
            joinedHit.setField("tagStart", bHit.getField("start"));
            joinedHit.setField("tagEnd", bHit.getField("end"));
            joinedHits.add(joinedHit);
        }
        return joinedHits;
    }

    private Result joinWithRangeItemPerHit(Query query, Result aResult, JoinSpec spec, Execution execution) {
        OrItem rangesRoot = new OrItem();
        for (Hit hit : aResult.hits()) {
            if ( ! hit.isAuxiliary()) {
                rangesRoot.addItem(toRangeItem(hit));
            }
        }
        spec.addBTerms(query);
        query.getModel().getQueryTree().and(rangesRoot);

        Result result = execution.search(query);
        execution.fill(result, ATTRIBUTE_ONLY_SUMMARY_CLASS);
        for (Hit hit : aResult.hits()) {
            if (hit.isAuxiliary()) {
                result.hits().add(hit);
            }
        }
        return result;
    }

    private Item toRangeItem(Hit aHit) {
        AndItem rangeItem = new AndItem();
        rangeItem.addItem(new WordItem(aHit.getField("id").toString(), "id"));
        rangeItem.addItem(new RangeItem(Limit.NEGATIVE_INFINITY, new Limit((long) aHit.getField("end"), true), "start"));
        rangeItem.addItem(new RangeItem(new Limit((long) aHit.getField("start"), true), Limit.POSITIVE_INFINITY, "end"));
        return rangeItem;
    }

    private Result joinInContainer(Query query, Result aResult, JoinSpec spec, Execution execution) {
        Result result = new Result(query);
        Map<String, List<HitInterval>> hitsById = new LinkedHashMap<>();
        for (Hit hit : aResult.hits()) {
            if (hit.isAuxiliary()) {
                result.hits().add(hit);
            }
            else {
                hitsById.computeIfAbsent(hit.getField("id").toString(), __ -> new ArrayList<>())
                        .add(new HitInterval(hit));
            }
        }

        spec.addBTerms(query);
        hitsById.forEach((id, intervals) -> {
            for (Hit joinedHit : joinInContainer(query.clone(), id, intervals, execution)) {
                result.hits().add(joinedHit);
            }
        });
        return result;
    }

    private List<Hit> joinInContainer(Query query, String id, List<HitInterval> aIntervals, Execution execution) {
        query.getModel().getQueryTree().and(new WordItem(id, "id"));
        Result bResult = execution.search(query);
        execution.fill(bResult, ATTRIBUTE_ONLY_SUMMARY_CLASS);

        List<HitInterval> bIntervals = new ArrayList<>();
        for (Hit bHit : bResult.hits()) {
            if ( ! bHit.isAuxiliary()) {
                bIntervals.add(new HitInterval(bHit));
            }
        }

        aIntervals.sort(Comparator.comparingLong(Interval::start));
        bIntervals.sort(Comparator.comparingLong(Interval::start));

        List<Hit> joinedResults = new ArrayList<>();
        for (Intersection<HitInterval, HitInterval> intersection : Intersector.intersectX(aIntervals, bIntervals)) {
            Hit joinedHit = intersection.first().hit.clone();
            joinedHit.setField("tagid", intersection.second().hit.getField("tagid"));
            joinedHit.setField("tagStart", intersection.second().start());
            joinedHit.setField("tagEnd", intersection.second().end());
            joinedResults.add(joinedHit);
        }
        return joinedResults;
    }

    static class JoinSpec {

        enum Variant {
            /** Do a sub-query per hit for query a, based on query b. */
            queryPerHit,
            /** Combine all hits from query a to an additional filter for query b. */
            rangeItemPerHit,
            /** Get all hits for queries a and b, and combine the results in a searcher. */
            containerIntersect;

            static Variant fromString(String value) {
                if (value == null) return queryPerHit;
                switch (value) {
                    case "queryPerHit": return queryPerHit;
                    case "rangeItemPerHit": return rangeItemPerHit;
                    case "containerIntersect": return containerIntersect;
                    default: throw new IllegalArgumentException("Must specify variant= one of " + List.of(values()));
                }
            }
        }

        final Variant variant;

        final String aType;
        final String aField;
        final String aValue;
        final long aStart, aEnd;

        final String bType;
        final String bField;
        final String bValue;

        public JoinSpec(Variant variant, String aType, String aField, String aValue, long aStart, long aEnd,
                        String bType, String bField, String bValue) {
            this.variant = variant;
            this.aType = aType;
            this.aField = aField;
            this.aValue = aValue;
            this.aStart = aStart;
            this.aEnd = aEnd;
            this.bType = bType;
            this.bField = bField;
            this.bValue = bValue;
        }

        public Query addATerms(Query query) {
            query.getModel().setSources(aType);
            query.getModel().getQueryTree().and(new WordItem(aValue, aField));
            return addInterval(aStart, aEnd, query);
        }

        public Query addBTerms(Query query) {
            query.getModel().setSources(bType);
            query.getModel().getQueryTree().and(new WordItem(bValue, bField));
            return query;
        }

        public Query addInterval(long start, long end, Query query) {
            query.getModel().getQueryTree().and(new RangeItem(Limit.NEGATIVE_INFINITY, new Limit(end, true), "start"));
            query.getModel().getQueryTree().and(new RangeItem(new Limit(start, true), Limit.POSITIVE_INFINITY, "end"));
            return query;
        }

        static Optional<JoinSpec> from(Query query) {
            Variant variant = Variant.fromString(query.properties().getString("variant"));
            String aType = query.properties().getString("a.type");
            String aField = query.properties().getString("a.field");
            String aValue = query.properties().getString("a.value");
            Long aStart = query.properties().getLong("a.start");
            Long aEnd = query.properties().getLong("a.end");
            String bType = query.properties().getString("b.type");
            String bField = query.properties().getString("b.field");
            String bValue = query.properties().getString("b.value");

            if (aType == null || aField == null || aValue == null || aStart == null || aEnd == null ||
                bType == null || bField == null || bValue == null)
                return Optional.empty();
            return Optional.of(new JoinSpec(variant, aType, aField, aValue, aStart, aEnd, bType, bField, bValue));
        }

    }


    static class HitInterval implements Interval {

        private final Hit hit;

        HitInterval(Hit hit) {
            this.hit = hit;
        }

        @Override
        public long start() {
            return (long) hit.getField("start");
        }

        @Override
        public long end() {
            return (long) hit.getField("end");
        }

    }

}
