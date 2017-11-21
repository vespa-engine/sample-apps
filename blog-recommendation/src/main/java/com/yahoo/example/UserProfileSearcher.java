// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.example;

import com.yahoo.data.access.Inspectable;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.processing.request.CompoundName;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.searchchain.SearchChain;
import com.yahoo.tensor.Tensor;

import java.util.Iterator;

public class UserProfileSearcher extends Searcher {

    public Result search(Query query, Execution execution) {
        Object userIdProperty = query.properties().get("user_id");
        if (userIdProperty != null) {
            Hit userProfile = retrieveUserProfile(userIdProperty.toString(), execution);
            if (userProfile != null) {
                addUserProfileTensorToQuery(query, userProfile);
                addReadItemsToQuery(query, userProfile);
            }
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

        execution.fill(result); // this is needed to get the actual summary data

        Iterator<Hit> hiterator = result.hits().deepIterator();
        return hiterator.hasNext() ? hiterator.next() : null;
    }

    private void addReadItemsToQuery(Query query, Hit userProfile) {
        Object readItems = userProfile.getField("has_read_items");
        if (readItems != null && readItems instanceof Inspectable) {
            query.properties().set(new CompoundName("has_read_items"), readItems);
        }
    }

    private void addUserProfileTensorToQuery(Query query, Hit userProfile) {
        Object userItemCf = userProfile.getField("user_item_cf");
        if (userItemCf != null && userItemCf instanceof Tensor) {
            query.properties().set(new CompoundName("user_item_cf"), (Tensor)userItemCf);
        }
    }
}
