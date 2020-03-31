// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.cord19.searcher;

/**
 * This searcher fetches an article by looking up the id as passed by &id parameter
 * and extracts the scibert-nli sentence embeddings for title and abstract.
 *
 * It performs a searching using OR (nearestNeighbor(title_embeding) nearestNeighbor(abstrac_embedding))
 * AND filters with the regular query (subject to &type and &query or &yql)
 *
 * The ranking profile used to rank is <i>related-ann</i>
 */

import com.google.inject.Inject;
import com.yahoo.prelude.query.*;
import com.yahoo.processing.request.CompoundName;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.Hit;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.language.Linguistics;
import com.yahoo.tensor.Tensor;


public class RelatedPaperSearcherANN extends Searcher {

    public static CompoundName INPUT_ID = new CompoundName("id");
    public static CompoundName USE_ABSTRACT = new CompoundName("use-abstract");
    public static CompoundName REMOVE_ARTICLE_FROM_VIEW =  new CompoundName("remove-id");

    private Linguistics linguistics;
    public static String SUMMARY = "attributeprefetch";

    @Inject
    public RelatedPaperSearcherANN(Linguistics linguistics) {
        this.linguistics = linguistics;
    }

    class Article {
        Tensor title; //scibert-nli embedding representation
        Tensor article_abstract; //scibert-nli embedding representation

        Article(Tensor title, Tensor article_abstract) {
            this.title = title;
            this.article_abstract = article_abstract;
        }
    }

    @Override
    public Result search(Query query, Execution execution) {
        Integer id = query.properties().getInteger(INPUT_ID, null);
        boolean includeAbstract = query.properties().getBoolean(USE_ABSTRACT,false);
        boolean removeArticle = query.properties().getBoolean(REMOVE_ARTICLE_FROM_VIEW,true);

        if (id == null) {
            Result empty = new Result(query);
            empty.hits().addError(ErrorMessage.createBadRequest("No id parameter"));
            return empty;
        }

        Item queryTreeRoot = query.getModel().getQueryTree().getRoot();
        String summary = query.getPresentation().getSummary();

        query.getPresentation().setSummary(SUMMARY);
        WordItem idFilter = new WordItem(id.toString(), "id", true);
        query.getModel().getQueryTree().setRoot(idFilter);

        Result result = execution.search(query);
        execution.fill(result,SUMMARY);
        Article article = extractFromResult(result);
        if (article == null) {
            return new Result(query);
        }

        Query relatedQuery = generateRelatedQuery(article,includeAbstract,queryTreeRoot);
        relatedQuery.getPresentation().setSummary(summary);
        relatedQuery.getPresentation().setBolding(false);
        relatedQuery.setHits(query.getHits());
        query.attachContext(relatedQuery);
        if(removeArticle) {
            NotItem notItem = new NotItem();
            notItem.addPositiveItem(relatedQuery.getModel().getQueryTree().getRoot());
            notItem.addNegativeItem(idFilter);
            relatedQuery.getModel().getQueryTree().setRoot(notItem);
        }
        return execution.search(relatedQuery);
    }

    private Article extractFromResult(Result r) {
        if (r.getTotalHitCount() == 0 || r.hits().get(0) == null)
            return null;
        Hit hit = r.hits().get(0);
        Tensor title = (Tensor)hit.getField("title_embedding");
        Tensor article_tensor = (Tensor)hit.getField(("abstract_embedding"));
        return new Article(title,article_tensor);
    }

    /**
     *
     * @param a the article to fetch related articles for
     * @param includeAbstract if true also the abstract embedding is used
     * @param originalQueryRoot the original query tree root, to preserve filtering
     * @return The related query
     */

    private Query generateRelatedQuery(Article a, boolean includeAbstract, Item originalQueryRoot) {
        Query relatedQuery = new Query();
        Item nnRoot;
        NearestNeighborItem nnTitle = new NearestNeighborItem("title_embedding",
                "vector");
        nnTitle.setAllowApproximate(false);
        nnTitle.setTargetNumHits(10);
        if(includeAbstract && a.article_abstract != null) {
            NearestNeighborItem nnAbstract = new NearestNeighborItem("abstract_embedding",
                    "vector");
            nnAbstract.setAllowApproximate(false);
            nnAbstract.setTargetNumHits(10);
            nnRoot = new OrItem();
            ((OrItem) nnRoot).addItem(nnAbstract);
            ((OrItem) nnRoot).addItem(nnTitle);
        } else  {
            nnRoot = nnTitle;
        }
        AndItem finalQueryTree = new AndItem();
        finalQueryTree.addItem(nnRoot);
        finalQueryTree.addItem(originalQueryRoot);

        relatedQuery.getRanking().getFeatures().put("query(vector)", a.title);
        relatedQuery.getRanking().setProfile(("related-ann"));
        relatedQuery.getModel().getQueryTree().setRoot(finalQueryTree);
        return relatedQuery;
    }

}
