// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.example;

import com.yahoo.component.chain.dependencies.After;
import com.yahoo.data.access.Inspectable;
import com.yahoo.data.access.Inspector;
import com.yahoo.prelude.query.IntItem;
import com.yahoo.prelude.query.NotItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.processing.request.CompoundName;
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
                addUserProfileTensorToQuery(query, userProfile);

                NotItem notItem = new NotItem();
                notItem.addItem(new IntItem(1, "has_user_item_cf"));
                for (String item : getReadItems(userProfile.getField("has_read_items"))){
                    notItem.addItem(new WordItem(item, "post_id"));
                }
                query.getModel().getQueryTree().and(notItem);
            }
        }

        // Restrict to search in blog_posts
        query.getModel().setRestrict("blog_post");

        // Rank blog posts using tensor rank profile
        if(query.properties().get("ranking") == null) {
            query.properties().set(new CompoundName("ranking"), "tensor");
        }

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

    private void addUserProfileTensorToQuery(Query query, Hit userProfile) {
        Object userItemCf = userProfile.getField("user_item_cf");
        if (userItemCf != null && userItemCf instanceof Tensor) {
            query.getRanking().getFeatures().put("query(user_item_cf)", (Tensor)userItemCf);
        }
    }

    private List<String> getReadItems(Object readItems) {
        List<String> items = new ArrayList<>();
        if (readItems instanceof Inspectable) {
            for (Inspector entry : ((Inspectable)readItems).inspect().entries()) {
                items.add(entry.asString());
            }
        }
        return items;
    }
}
