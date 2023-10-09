// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.component.chain.dependencies.After;
import com.yahoo.prelude.query.AndItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.WeightedSetItem;
import com.yahoo.processing.request.CompoundName;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;

/**
 * Searcher which accepts a input set filter which which bypass YQL parsing of large set
 * filters https://docs.vespa.ai/en/performance/feature-tuning.html#multi-lookup-set-filtering
 *
 * Example usage
 *
 * /search/?yql=select * from music where userQuery()&
 * set-filter=2015,2016,2018&
 * set-filter-field-name=year&query=artist:metallica
 *
 * The set filter is ANDed with the YQL parts of the query
 *
 * Above example is turned into
 * select * from music where (artist contains "metallica") AND
 * ({"filter": true, "ranked": false}weightedSet(year, {"2015": 1, "2016": 1, "2018": 1}))
 *
 */

@After("ExternalYql")
public class SetFilterSearcher extends Searcher {

    static CompoundName setFilterValues = new CompoundName("set-filter");
    static CompoundName setFilterFieldName = new CompoundName("set-filter-field-name");

    @Override
    public Result search(Query query, Execution execution) {
        String fieldName = query.properties().getString(setFilterFieldName);
        String values = query.properties().getString(setFilterValues);
        if(fieldName == null || values == null) {
            return execution.search(query);
        }
        String[] filters = values.split(",");
        WeightedSetItem setItem = new WeightedSetItem(fieldName);
        setItem.setFilter(true);
        setItem.setRanked(false);
        for(String f: filters) {
            setItem.addToken(f);
        }
        Item queryRoot = query.getModel().getQueryTree().getRoot();
        if (queryRoot instanceof AndItem) {
            ((AndItem) queryRoot).addItem(setItem);
        } else {
            AndItem and = new AndItem();
            and.addItem(queryRoot);
            and.addItem(setItem);
            query.getModel().getQueryTree().setRoot(and);
        }
        return execution.search(query);
    }
}
