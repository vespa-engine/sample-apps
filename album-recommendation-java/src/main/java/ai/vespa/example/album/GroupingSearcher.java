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

public class GroupingSearcher extends Searcher {
    public final static String GROUPING_FIELD = "year";
    private static final Logger log = Logger.getLogger(MetalSearcher.class.getName());

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
