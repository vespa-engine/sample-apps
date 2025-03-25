// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.shopping;

import com.yahoo.component.annotation.Inject;
import com.yahoo.language.Language;
import com.yahoo.language.Linguistics;
import com.yahoo.language.process.LinguisticsParameters;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.prelude.query.PrefixItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.prelude.query.FuzzyItem;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;

import java.util.ArrayList;
import java.util.List;

/**
 * Searches for suggestions using a combination of prefix matching and token based matching
 */

public class SuggestionSearcher extends Searcher {

    private static final String SUGGESTION_SUMMARY = "query";
    private static final String SUGGESTION_RANK_PROFILE = "default";


    private final Linguistics linguistics;
    @Inject
    public SuggestionSearcher(Linguistics linguistics) {
        this.linguistics = linguistics;
    }

    @Override
    public Result search(Query query, Execution execution) {
        String userQuery = query.properties().getString("term");
        if (userQuery == null) return execution.search(query);
        List<String> tokens = tokenize(userQuery);
        return getSuggestions(userQuery, tokens, execution, query);
    }

    private List<String> tokenize(String userQuery) {
        List<String> result = new ArrayList<>(6);
        Iterable<Token> tokens = this.linguistics.getTokenizer()
                                         .tokenize(userQuery, new LinguisticsParameters(Language.fromLanguageTag("en"), StemMode.NONE, false, true));
        for(Token t: tokens) {
            if (t.isIndexable())
                result.add(t.getTokenString());
        }
        return result;
    }

    private Result getSuggestions(String userQuery, List<String> tokens, Execution execution, Query originalQuery) {
        Query query = new Query();
        query.getPresentation().setSummary(SUGGESTION_SUMMARY);
        originalQuery.attachContext(query);
        query.setHits(5);
        query.getModel().setRestrict("query");
        query.getRanking().setProfile(SUGGESTION_RANK_PROFILE);

        Item suggestionQuery = buildSuggestionQueryTree(userQuery, tokens);
        query.getModel().getQueryTree().setRoot(suggestionQuery);
        Result suggestionResult = execution.search(query);
        execution.fill(suggestionResult, SUGGESTION_SUMMARY);
        return suggestionResult;
    }

    private Item buildSuggestionQueryTree(String userQuery, List<String> tokens) {
        PrefixItem prefix = new PrefixItem(userQuery, "default");
        OrItem relaxedMatching = new OrItem();
        for(String t: tokens) {
            int length = t.length();
            if(length <= 3) {
             WordItem word = new WordItem(t, "words", true);
             relaxedMatching.addItem(word);
            } else {
                int maxDistance = 1;
                if (length > 6)
                    maxDistance = 2;
                FuzzyItem fuzzyItem = new FuzzyItem("words",
                        true, t, maxDistance, 2);
                relaxedMatching.addItem(fuzzyItem);
            }
        }
        if(relaxedMatching.getItemCount() == 0)
            return prefix;
        OrItem orItem = new OrItem();
        orItem.addItem(prefix);
        orItem.addItem(relaxedMatching);
        return orItem;
    }
}
