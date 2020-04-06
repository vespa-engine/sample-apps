// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19.searcher;

import com.google.inject.Inject;
import com.yahoo.language.Language;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.process.TokenType;
import com.yahoo.language.process.Tokenizer;
import com.yahoo.prelude.query.AndItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.WeakAndItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.language.Linguistics;
import java.util.Set;

/**
 * Fetches related articles by searching for articles with many of the same words using a WeakAnd item.
 *
 * @author jobergum
 */
public class RelatedArticlesByWeakAndSearcher extends RelatedArticlesSearcher {

    private final Linguistics linguistics;
    public static final String summary = "full";

    @Inject
    public RelatedArticlesByWeakAndSearcher(Linguistics linguistics) {
        this.linguistics = linguistics;
    }

    @Override
    protected void addRelatedItem(Integer relatedArticleId, boolean includeAbstract, Execution execution, Query query) {
        Article article = fetchArticle(relatedArticleId, execution, query);
        addWeakAndItem(article, includeAbstract, query);
    }

    private Article fetchArticle(Integer id, Execution execution, Query query) {
        Query articleQuery = new Query();
        articleQuery.attachContext(query);
        articleQuery.setHits(1);
        articleQuery.getPresentation().setSummary(summary);
        WordItem idFilter = new WordItem(id.toString(), "id", true);
        articleQuery.getModel().getQueryTree().setRoot(idFilter);

        Result result = execution.search(articleQuery);
        execution.fill(result, summary);
        return articleFrom(result);
    }

    private Article articleFrom(Result result) {
        if (result.hits().size() < 1)
            throw new IllegalArgumentException("Requested article not found");

        Hit hit = result.hits().get(0);
        return new Article((String)hit.getField("title-full"),
                           (String)hit.getField("abstract-full"));
    }

    private Iterable<Token> tokenize(String data) {
        Tokenizer tokenizer = this.linguistics.getTokenizer();
        return tokenizer.tokenize(data,
                Language.ENGLISH, StemMode.NONE, true);
    }

    private void addWeakAndItem(Article article, boolean includeAbstract, Query query) {
        WeakAndItem weakAndItem = new WeakAndItem();
        if (article.title != null) {
            for (Token t : tokenize(article.title)) {
                if (!stopwords.contains((t.getTokenString())) &&
                    (t.getType() == TokenType.ALPHABETIC || t.getType() == TokenType.NUMERIC)) {
                    WordItem tokenItem = new WordItem(t.getTokenString(), "default", true);
                    tokenItem.setWeight(150);
                    weakAndItem.addItem(tokenItem);
                }
            }
        }
        if (article.articleAbstract != null && includeAbstract)  {
            for (Token t : tokenize(article.articleAbstract)) {
                if (!stopwords.contains((t.getTokenString())) && (t.getType() == TokenType.ALPHABETIC || t.getType() == TokenType.NUMERIC)) {
                    WordItem tokenItem = new WordItem(t.getTokenString(), "default", true);
                    tokenItem.setWeight(100);
                    weakAndItem.addItem(tokenItem);
                }
            }
        }

        // Combine
        Item root = query.getModel().getQueryTree().getRoot();
        AndItem andItem = new AndItem();
        andItem.addItem(root);
        andItem.addItem(weakAndItem);
        query.getModel().getQueryTree().setRoot(andItem);
    }

    private static Set<String> stopwords = Set.of(
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
                "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
                "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
                "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
                "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now");

    private static class Article {

        final String title;
        final String articleAbstract;

        Article(String title, String articleAbstract) {
            this.title = title;
            this.articleAbstract = articleAbstract;
        }
    }

}
