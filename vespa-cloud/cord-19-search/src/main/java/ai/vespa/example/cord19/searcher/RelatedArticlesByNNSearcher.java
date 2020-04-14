// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19.searcher;

import com.yahoo.prelude.query.AndItem;
import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.prelude.query.TermItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;

/**
 * Fetches related articles by searching for articles with many of the same words using a WeakAnd item.
 *
 * @author jobergum
 */
public class RelatedArticlesByNNSearcher extends RelatedArticlesSearcher {

    private static String tensorSummary = "attributeprefetch";

    @Override
    protected void addRelatedItem(Integer relatedArticleId, boolean includeAbstract, Execution execution, Query query) {
        Article article = fetchArticle(relatedArticleId, execution, query);
        addANNItem(article, includeAbstract, query);
    }

    private Article fetchArticle(Integer id, Execution execution, Query query) {
        Query articleQuery = new Query();
        query.attachContext(articleQuery);
        articleQuery.getPresentation().setSummary(tensorSummary);
        WordItem idFilter = new WordItem(id.toString(), "id", true);
        articleQuery.getModel().getQueryTree().setRoot(idFilter);
        articleQuery.getModel().setRestrict("doc");
        articleQuery.setHits(1);
        articleQuery.getRanking().setProfile("unranked");
        Result articleResult = execution.search(articleQuery);
        execution.fill(articleResult, tensorSummary);
        return articleFrom(articleResult);
    }

    private Article articleFrom(Result result) {
        if (result.hits().size() < 1)
            throw new IllegalArgumentException("Requested article not found");

        Hit articleHit = result.hits().get(0);
        return new Article((Tensor) articleHit.getField("title_embedding"),
                (Tensor) articleHit.getField("abstract_embedding"),
                (Tensor) articleHit.getField("specter_embedding"));
    }

    /**
     * Adds aterm to the given query to find related articles
     *
     * @param article         the article to fetch related articles for
     * @param includeAbstract whether the vector embedding from the abstract should be used
     */
    private void addANNItem(Article article, boolean includeAbstract, Query query) {
        Item nnRoot;
        String rankProfile = "related-ann";
        if(query.properties().getBoolean("use-specter")) {
            nnRoot = createNNItem("specter_embedding", "specter_vector");
            query.getRanking().getFeatures().put("query(specter_vector)", article.specterEmbedding);
            rankProfile = "related-specter";
        } else {
            Item nnTitle = createNNItem("title_embedding", "title_vector");
            query.getRanking().getFeatures().put("query(title_vector)", article.titleEmbedding);
            if (includeAbstract && article.abstractEmbedding != null) {
                NearestNeighborItem nnAbstract = createNNItem("abstract_embedding", "abstract_vector");
                query.getRanking().getFeatures().put("query(abstract_vector)", article.abstractEmbedding);
                nnRoot = new OrItem();
                ((OrItem) nnRoot).addItem(nnAbstract);
                ((OrItem) nnRoot).addItem(nnTitle);
            } else {
                nnRoot = nnTitle;
            }
        }
        filter(rankProfile, query,nnRoot);
    }

    private void filter(String rankprofile, Query query, Item nn) {
        Item root = query.getModel().getQueryTree().getRoot();
        if (!hasTextTerms(root)) {
            query.getModel().getQueryTree().setRoot(nn);
            // query is empty -> Must rank by vectors
            query.getRanking().setProfile(rankprofile);
        } else {
            AndItem andItem = new AndItem();
            andItem.addItem(root);
            andItem.addItem(nn);
            query.getModel().getQueryTree().setRoot(andItem);
        }
    }

    private NearestNeighborItem createNNItem(String field, String query) {
        NearestNeighborItem nnTitle = new NearestNeighborItem(field, query);
        nnTitle.setAllowApproximate(false);
        nnTitle.setTargetNumHits(100);
        return nnTitle;
    }

    private boolean hasTextTerms(Item item) {
        if (item instanceof CompositeItem) {
            for (Item child : ((CompositeItem) item).items())
                if (hasTextTerms(child))
                    return true;

        }
        if ((item instanceof TermItem) && !item.isFilter())
            return true;
        return false;
    }

    private static class Article {

        final Tensor titleEmbedding; // scibert-nli embedding
        final Tensor abstractEmbedding; // scibert-nli embedding
        final Tensor specterEmbedding; //SPECTER EMBEDDING

        Article(Tensor titleEmbedding, Tensor abstractEmbedding, Tensor specterEmbedding) {
            this.titleEmbedding = titleEmbedding;
            this.abstractEmbedding = abstractEmbedding;
            this.specterEmbedding = specterEmbedding;
        }

    }

}
