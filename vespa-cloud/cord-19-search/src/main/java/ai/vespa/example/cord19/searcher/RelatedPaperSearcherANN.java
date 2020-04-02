// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19.searcher;

import com.yahoo.prelude.query.AndItem;
import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.IntItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.prelude.query.NotItem;
import com.yahoo.prelude.query.OrItem;
import com.yahoo.prelude.query.TermItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.processing.request.CompoundName;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.Hit;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;

import java.util.Optional;

/**
 * This searcher fetches an article by looking up the id as passed by &id parameter
 * and extracts the scibert-nli sentence embeddings for title and abstract.
 *
 * It performs a searching using OR (nearestNeighbor(title_embeding) nearestNeighbor(abstrac_embedding))
 * AND filters with the regular query (subject to &type and &query or &yql)
 *
 * The ranking profile used to rank is <i>related-ann</i>
 *
 * @author jobergum
 */
public class RelatedPaperSearcherANN extends Searcher {

    private static String relatedToField = "related_to";
    private static String tensorSummary = "attributeprefetch";

    private static CompoundName inputId = new CompoundName("id");
    private static CompoundName useAbstractEmbedding = new CompoundName("use-abstract");
    private static CompoundName removeArticleFromResult =  new CompoundName("remove-id");

    @Override
    public Result search(Query query, Execution execution) {
        boolean includeAbstract = query.properties().getBoolean(useAbstractEmbedding, false);
        boolean filterRelatedArticle = query.properties().getBoolean(removeArticleFromResult, true);

        Optional<Integer> relatedArticleId = relatedArticleIdFrom(query);
        if (relatedArticleId.isEmpty()) return execution.search(query);

        Article article = fetchArticle(relatedArticleId.get(), execution, query);
        addANNTerm(article, includeAbstract, query);

        if (filterRelatedArticle)
            addArticleFilterTerm(relatedArticleId.get(), query);
        return execution.search(query);
    }

    private Optional<Integer> relatedArticleIdFrom(Query query) {
        Optional<Integer> idTerm = extractRelatedToItem(query.getModel().getQueryTree());
        Optional<Integer> idParameter = Optional.ofNullable(query.properties().getInteger(inputId));
        if (idTerm.isPresent() && idParameter.isPresent() && ! idTerm.equals(idParameter))
            throw new IllegalArgumentException("Cannot specify both a related article id parameter and id term");
        return idTerm.isPresent() ? idTerm : idParameter;
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
        return articleFromResult(articleResult);
    }

    /**
     * Finds a related_to item in the query and removes it
     *
     * @return the id specified by the related_to item, or empty if none
     */
    private Optional<Integer> extractRelatedToItem(CompositeItem parent) {
        for (Item child : parent.items()) {
            if (child instanceof IntItem) {
                IntItem intItem = (IntItem) child;
                if (intItem.getIndexName().equals(relatedToField)) {
                    parent.removeItem(child);
                    return Optional.of(Integer.parseInt(intItem.getRawWord()));
                }
            } else if (child instanceof CompositeItem) {
                Optional<Integer> relatedId = extractRelatedToItem((CompositeItem) child);
                if (relatedId.isPresent())
                    return relatedId;
            }
        }
        return Optional.empty();
    }

    private Article articleFromResult(Result result) {
        if (result.hits().size() < 1)
            throw new IllegalArgumentException("Requested article not found");

        Hit articleHit = result.hits().get(0);
        return new Article((Tensor)articleHit.getField("title_embedding"),
                           (Tensor)articleHit.getField(("abstract_embedding")));
    }

    /**
     * Adds aterm to the given query to find related articles
     *
     * @param article the article to fetch related articles for
     * @param includeAbstract whether the vector embedding from the abstract should be used
     */
    private void addANNTerm(Article article, boolean includeAbstract, Query query) {
        Item nnTitle = createNNItem("title_embedding", "title_vector");
        query.getRanking().getFeatures().put("query(title_vector)", article.titleEmbedding);

        Item nnRoot;
        if (includeAbstract && article.abstractEmbedding != null) {
            NearestNeighborItem nnAbstract = createNNItem("abstract_embedding", "abstract_vector");
            query.getRanking().getFeatures().put("query(abstract_vector)", article.abstractEmbedding);
            nnRoot = new OrItem();
            ((OrItem) nnRoot).addItem(nnAbstract);
            ((OrItem) nnRoot).addItem(nnTitle);
        } else  {
            nnRoot = nnTitle;
        }

        // Combine
        Item root = query.getModel().getQueryTree().getRoot();
        if ( ! hasTextTerms(root)) {
            query.getModel().getQueryTree().setRoot(nnRoot);
            // query is empty -> Must rank by vectors
            query.getRanking().setProfile("related-ann");
        } else {
            AndItem andItem = new AndItem();
            andItem.addItem(root);
            andItem.addItem(nnRoot);
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
            for (Item child : ((CompositeItem)item).items())
                if (hasTextTerms(child))
                    return true;

        }
        if ((item instanceof TermItem) && ! item.isFilter())
            return true;
        return false;
    }

    private void addArticleFilterTerm(Integer relatedArticleId, Query query) {
        NotItem notItem = new NotItem();
        notItem.addPositiveItem(query.getModel().getQueryTree().getRoot());
        notItem.addNegativeItem(new WordItem(relatedArticleId.toString(), "id", true));
        query.getModel().getQueryTree().setRoot(notItem);
    }

    private static class Article {

        final Tensor titleEmbedding; // scibert-nli embedding
        final Tensor abstractEmbedding; // scibert-nli embedding

        Article(Tensor titleEmbedding, Tensor abstractEmbedding) {
            this.titleEmbedding = titleEmbedding;
            this.abstractEmbedding = abstractEmbedding;
        }

    }

}
