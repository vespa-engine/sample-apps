// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example;

import com.yahoo.prelude.query.WordItem;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;


public class UserProfileSearcher extends Searcher {

    public Result search(Query query, Execution execution) {

        // Get tensor and read items from user profile
        Object userIdProperty = query.properties().get("user_id");
        if (userIdProperty != null) {

            // Retrieve user embedding by doing a search for the user_id and extract the tensor
            Tensor userEmbedding = retrieveUserEmbedding(userIdProperty.toString(), execution);

            // Create a new search using the user's embedding tensor
            NearestNeighborItem nn = new NearestNeighborItem("embedding", "user_embedding");
            nn.setTargetNumHits(query.getHits());
            nn.setAllowApproximate(true);

            query.getModel().getQueryTree().setRoot(nn);
            query.getRanking().getFeatures().put("query(user_embedding)", userEmbedding);
            query.getRanking().setProfile("recommendation");
            query.getModel().setRestrict("news");
        }

        return execution.search(query);
    }

    private Tensor retrieveUserEmbedding(String userId, Execution execution) {
        Query query = new Query();
        query.getModel().setRestrict("user");
        query.getModel().getQueryTree().setRoot(new WordItem(userId, "user_id"));
        query.setHits(1);

        Result result = execution.search(query);
        execution.fill(result); // This is needed to get the actual summary data

        if (result.getTotalHitCount() == 0)
            throw new RuntimeException("User id " + userId + " not found...");
        return (Tensor) result.hits().get(0).getField("embedding");
    }

}
