// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;


import com.yahoo.prelude.query.WeakAndItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;


public class RetrievalModelSearcher extends Searcher {

    @Override
    public Result search(Query query, Execution execution) {
        String queryInput = query.getModel().getQueryString();
        if (query.getModel().getQueryString() == null ||
                query.getModel().getQueryString().length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));


        String wandField = query.properties().getString("wand.field", "default");
        int wandHits = query.properties().getInteger("wand.hits", query.getHits());
        query.getModel().getQueryTree().setRoot(sparseRetrieval(queryInput, query, wandField, wandHits));
        int reRankHits = query.properties().getInteger("phase.count", 1000);
        query.getRanking().getProperties().put("vespa.hitcollector.heapsize", reRankHits);
        return execution.search(query);
    }

    private WeakAndItem sparseRetrieval(String queryInput, Query query, String field, int hits) {
        String[] tokens = queryInput.split(" ");
        WeakAndItem wand = new WeakAndItem();
        wand.setN(hits);
        for (String t : tokens) {
            if (t.isBlank() || t.isEmpty())
                continue;
            wand.addItem(new WordItem(t, field, true));
        }
        return wand;
    }

}
