// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.grouping.request.*;
import com.yahoo.search.grouping.GroupingRequest;
import com.yahoo.search.grouping.result.Group;
import com.yahoo.search.grouping.result.GroupList;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;

import java.util.logging.Logger;

/**
 * A searcher which adds an grouping to queries.
 *
 * See https://docs.vespa.ai/en/searcher-development.html
 * See https://docs.vespa.ai/en/grouping.html#global-limit-for-grouping-queries
 */
public class GroupingSearcher extends Searcher {
    public final static String GROUPING_FIELD = "year";
    private static final Logger log = Logger.getLogger(MetalSearcher.class.getName());

    /**
     * Performs a search and post-processes the result with a grouping operation based on a specific attribute.
     * <p>
     * This method uses Vespa to count the number of results per unique value of
     * the {@code GROUPING_FIELD}. It constructs a grouping request that groups hits by the specified field
     * and counts how many documents fall into each group. These aggregated counts are then added back as
     * synthetic hits in the final {@link Result} object.
     * </p>
     *
     * Args:
     *     query (Query): The search query.
     *     execution (Execution): The Vespa execution context used to perform the search.
     *
     * Returns:
     *     Result: The original search result with additional hits representing grouped counts by the given field.
     */
    @Override
    public Result search(Query query, Execution execution) {
        GroupingRequest grpRequest = GroupingRequest.newInstance(query);
        grpRequest.setRootOperation(new AllOperation()
                .setGroupBy(new AttributeValue(GROUPING_FIELD))
                .addChild(new EachOperation()
                        .addOutput(new CountAggregator().setLabel("count"))));

        Result result = execution.search(query);

        Group root = grpRequest.getResultGroup(result);
        GroupList yearGroups = root.getGroupList(GROUPING_FIELD);
        if (yearGroups == null) {
            return result;
        }
        for(Hit hit : yearGroups) {
           Group group = (Group) hit;
            Long count = (Long) group.getField("count");
            Hit metadata = new Hit(group.getGroupId().toString(), 1.0);
            metadata.setField("count", count);
            result.hits().add(metadata);
        }
        return result;
    }
}
