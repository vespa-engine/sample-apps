// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example;

import com.yahoo.component.chain.dependencies.After;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.SearchChain;
import com.yahoo.tensor.Tensor;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

@After("ExternalYql")
public class UserProfileSearcher extends Searcher {

    public Result search(Query query, Execution execution) {

        // Get tensor and read items from user profile
        Object userIdProperty = query.properties().get("user_id");
        if (userIdProperty != null) {
            Hit userProfile = retrieveUserProfile(userIdProperty.toString(), execution);
            if (userProfile != null) {

                // Consider adding items already read or filters? Show filters with ANNs pretty good

                Tensor userEmbedding = (Tensor) userProfile.getField("embedding");

                // Add user's embedding as a nearest neighbor item
                // TODO: compare brute force with approximate

                NearestNeighborItem nn = new NearestNeighborItem("embedding", "user_embedding");
                nn.setTargetNumHits(query.getHits());
                // nn.setAllowApproximate(true);
                query.getRanking().getFeatures().put("query(user_embedding)", userEmbedding);

                query.getModel().getQueryTree().setRoot(nn);
                query.getRanking().setProfile("dot_prod");
            }
        }

        // Restrict to search in news.
        query.getModel().setRestrict("news");

        return execution.search(query);
    }

    private Hit retrieveUserProfile(String userId, Execution execution) {
        Query query = new Query();
        query.getModel().setRestrict("user");
        query.getModel().getQueryTree().setRoot(new WordItem(userId, "user_id"));
        query.setHits(1);

        SearchChain vespaChain = execution.searchChainRegistry().getComponent("vespa");
        Result result = new Execution(vespaChain, execution.context()).search(query);

        execution.fill(result); // This is needed to get the actual summary data

        Iterator<Hit> hiterator = result.hits().deepIterator();
        return hiterator.hasNext() ? hiterator.next() : null;
    }

}
