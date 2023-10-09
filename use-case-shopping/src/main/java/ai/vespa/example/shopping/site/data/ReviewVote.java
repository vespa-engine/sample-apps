// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.data;

import java.io.IOException;
import java.util.Map;

public class ReviewVote {

    private final static String upvote = "{ \"fields\": { \"upvotes\": { \"increment\":1 } } }";
    private final static String downvote = "{ \"fields\": { \"downvotes\": { \"increment\":1 } } }";

    /**
     * Fires off a partial update of the review document to update the number of upvotes
     */
    public static void vote(SimpleHttpClient client, Map<String, String> properties) throws IOException {
        String item = properties.get("i");
        String reviewerId = properties.get("rid");
        String dir = properties.get("dir");
        if (item == null || item.length() == 0 || reviewerId == null || reviewerId.length() == 0 || dir == null || dir.length() == 0) {
            return;
        }
        String docid = item + "-" + reviewerId;
        String update = (Integer.parseInt(dir) > 0) ? upvote : downvote;
        String response = client.put("/document/v1/review/review/docid/" + docid, update);
        if (response == null) {
            throw new RuntimeException("Update had no return value");
        }

    }

}
