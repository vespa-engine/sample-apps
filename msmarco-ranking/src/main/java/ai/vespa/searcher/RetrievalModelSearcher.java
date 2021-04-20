// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;


import com.google.inject.Inject;
import com.yahoo.language.Language;
import com.yahoo.language.Linguistics;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.prelude.query.WeakAndItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import java.util.ArrayList;
import java.util.List;

public class RetrievalModelSearcher extends Searcher {

    private Linguistics linguistics;

    @Inject
    public RetrievalModelSearcher(Linguistics linguistics) {
        this.linguistics = linguistics;
    }

    @Override
    public Result search(Query query, Execution execution) {
        String queryInput = query.getModel().getQueryString();
        if (query.getModel().getQueryString() == null ||
                query.getModel().getQueryString().length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));

        String wandField = query.properties().getString("wand.field", "default");
        int wandHits = query.properties().getInteger("wand.hits", query.getHits());
        query.getModel().getQueryTree().setRoot(sparseRetrieval(queryInput, wandField, wandHits));
        int reRankHits = query.properties().getInteger("phase.count", 1000);
        query.getRanking().getProperties().put("vespa.hitcollector.heapsize", reRankHits);
        return execution.search(query);
    }

    private List<String> tokenize(String query) {
        Iterable<Token> tokens = this.linguistics.getTokenizer().
                tokenize(query, Language.ENGLISH, StemMode.NONE,true);
        List<String> queryTokens = new ArrayList<>();
        for(Token t:tokens) {
            if (t.isIndexable())
                queryTokens.add(t.getTokenString());
        }
        return queryTokens;
    }

    private WeakAndItem sparseRetrieval(String queryInput, String field, int hits) {
        WeakAndItem wand = new WeakAndItem();
        wand.setN(hits);
        for (String t : tokenize(queryInput)) {
            wand.addItem(new WordItem(t, field, true));
        }
        return wand;
    }

}
