// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.joins;

import com.yahoo.prelude.query.IntItem;
import com.yahoo.prelude.query.Limit;
import com.yahoo.prelude.query.NotItem;
import com.yahoo.prelude.query.RangeItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class JoinSearcher extends Searcher {

    @Override
    public Result search(Query query, Execution execution) {
        Optional<JoinSpec> joinSpec = JoinSpec.from(query);
        if (joinSpec.isEmpty()) return execution.search(query); // No join spec so skip this searcher

        Query inputQuery = query.clone();

        Result aResult = execution.search(joinSpec.get().addATerms(query));
        execution.fillAttributes(aResult);

        return join(inputQuery, aResult, joinSpec.get(), execution);
    }

    private Result join(Query query, Result aResult, JoinSpec spec, Execution execution) {
        Result result = new Result(query);
        for (Hit hit : aResult.hits()) {
            if (hit.isAuxiliary()) {
                result.hits().add(hit);
            }
            else {
                for (Hit joinedHit : joinWith(hit, query.clone(), spec, execution))
                    result.hits().add(joinedHit);
            }
        }
        return result;
    }

    private List<Hit> joinWith(Hit aHit, Query query, JoinSpec spec, Execution execution) {
        spec.addBTerms(query);
        query.getModel().getQueryTree().and(new WordItem(aHit.getField("id").toString(), "id"));
        spec.addInterval((long)aHit.getField("start"), (long)aHit.getField("end"), query);
        Result bResult = execution.search(query);
        execution.fillAttributes(bResult);
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

    static class JoinSpec {

        final String aType;
        final String aField;
        final String aValue;
        final long aStart, aEnd;

        final String bType;
        final String bField;
        final String bValue;

        public JoinSpec(String aType, String aField, String aValue, long aStart, long aEnd,
                        String bType, String bField, String bValue) {
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
            return Optional.of(new JoinSpec(aType, aField, aValue, aStart, aEnd, bType, bField, bValue));
        }

    }

}
