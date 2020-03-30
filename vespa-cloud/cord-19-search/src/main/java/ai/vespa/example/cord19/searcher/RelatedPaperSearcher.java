// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19.searcher;

import com.google.inject.Inject;
import com.yahoo.language.Language;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.process.TokenType;
import com.yahoo.language.process.Tokenizer;
import com.yahoo.prelude.query.WeakAndItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.processing.request.CompoundName;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.Hit;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.language.Linguistics;
import java.util.HashSet;
import java.util.Set;


public class RelatedPaperSearcher extends Searcher {

    public static CompoundName INPUT_ID = new CompoundName("id");
    public static CompoundName USE_ABSTRACT = new CompoundName("use-abstract");
    private Linguistics linguistics;
    public static String SUMMARY = "full";

    @Inject
    public RelatedPaperSearcher(Linguistics linguistics) {
        this.linguistics = linguistics;
    }

    class Article {
        String title;
        String article_abstract;

        Article(String title, String article_abstract) {
            this.title = title;
            this.article_abstract = article_abstract;
        }
    }

    @Override
    public Result search(Query query, Execution execution) {
        Integer id = query.properties().getInteger(INPUT_ID, null);
        boolean includeAbstract = query.properties().getBoolean(USE_ABSTRACT,false);

        if (id == null) {
            Result empty = new Result(query);
            empty.hits().addError(ErrorMessage.createBadRequest("No id parameter"));
            return empty;
        }
        query.getPresentation().setSummary(SUMMARY);
        WordItem idFilter = new WordItem(id.toString(), "id", true);
        query.getModel().getQueryTree().setRoot(idFilter);

        Result result = execution.search(query);
        execution.fill(result,SUMMARY);
        Article article = extractFromResult(result,query);
        if (article == null) {
           return new Result(query);
        }
        Query relatedQuery = generateRelatedQuery(article,includeAbstract);
        relatedQuery.getPresentation().setSummary(query.getPresentation().getSummary());
        relatedQuery.getPresentation().setBolding(false);
        relatedQuery.setHits(query.getHits());
        query.attachContext(relatedQuery);
        relatedQuery.getRanking().setProfile(query.getRanking().getProfile());
        return execution.search(relatedQuery);
    }

    private Article extractFromResult(Result r,Query query) {
        if (r.getTotalHitCount() == 0 || r.hits().get(0) == null)
            return null;
        Hit hit = r.hits().get(0);
        String title = (String) hit.getField("title-full");
        String hit_abstract = (String) hit.getField("abstract-full");
        return new Article(title,hit_abstract);
    }

    private Iterable<Token> tokenize(String data) {
        Tokenizer tokenizer = this.linguistics.getTokenizer();
        return tokenizer.tokenize(data,
                Language.ENGLISH, StemMode.NONE, true);
    }

    private Query generateRelatedQuery(Article a, boolean includeAbstract) {
        Query relatedQuery = new Query();
        WeakAndItem weakAndItem = new WeakAndItem();
        if(a.title != null) {
            for (Token t : tokenize(a.title)) {
                if (!stopwordSet.contains((t.getTokenString())) &&
                        (t.getType() == TokenType.ALPHABETIC || t.getType() == TokenType.NUMERIC)) {
                    WordItem tokenItem = new WordItem(t.getTokenString(), "default", true);
                    tokenItem.setWeight(150);
                    weakAndItem.addItem(tokenItem);
                }
            }
        }
        if(a.article_abstract != null && includeAbstract)  {
            for (Token t : tokenize(a.article_abstract)) {
                if (!stopwordSet.contains((t.getTokenString())) && (t.getType() == TokenType.ALPHABETIC || t.getType() == TokenType.NUMERIC)) {
                    WordItem tokenItem = new WordItem(t.getTokenString(), "default", true);
                    tokenItem.setWeight(100);
                    weakAndItem.addItem(tokenItem);
                }
            }
        }
        relatedQuery.getModel().getQueryTree().setRoot(weakAndItem);
        return relatedQuery;
    }
    private static Set<String> stopwordSet = new HashSet<>();
    static {
        String[] stopwords = new String[]{"i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
                "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
                "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
                "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
                "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"};
        for(String s: stopwords)
            stopwordSet.add(s);
    }
}
