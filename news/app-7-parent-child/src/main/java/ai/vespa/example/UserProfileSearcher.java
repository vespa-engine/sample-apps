// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example;

import com.yahoo.prelude.query.WordItem;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;


public class UserProfileSearcher extends Searcher {

    public Result search(Query query, Execution execution) {

        // Get tensor and read items from user profile
        Object userIdProperty = query.properties().get("user_id");
        if (userIdProperty != null) {

            // Retrieve user embedding by doing a search for the user_id and extract the tensor
            Tensor userEmbedding = retrieveUserEmbedding(userIdProperty.toString(), execution, query);

            // Create a new search using the user's embedding tensor
            NearestNeighborItem nn = new NearestNeighborItem("embedding", "user_embedding");
            nn.setTargetNumHits(query.getHits());
            nn.setAllowApproximate(true);

            query.getModel().getQueryTree().setRoot(nn);
            query.getRanking().getFeatures().put("query(user_embedding)", userEmbedding);
            query.getModel().setRestrict("news");

            // Override default ranking profile
            if (query.getRanking().getProfile().equals("default")) {
                query.getRanking().setProfile("recommendation");
            }
        }

        return execution.search(query);
    }

    private Tensor retrieveUserEmbedding(String userId, Execution execution,Query originalQuery) {
        Query query = new Query();
        originalQuery.attachContext(query);
        query.getModel().setRestrict("user");
        query.getModel().getQueryTree().setRoot(new WordItem(userId, "user_id"));
        query.setHits(1);
        query.getRanking().setProfile("single-phase-user-fetch");
        Result result = execution.search(query);
        if (result.getTotalHitCount() == 0)
            throw new RuntimeException("User id " + userId + " not found...");
        Hit hit = result.hits().get(0);
        FeatureData featureData = (FeatureData)hit.getField("matchfeatures");
        return featureData.getTensor("attribute(embedding)");
    }

}
