// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19.searcher;

import com.yahoo.prelude.query.AndItem;
import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.IntItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.prelude.query.NotItem;
import com.yahoo.prelude.query.NullItem;
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

    public static String RELATED_TO_FIELD = "related_to";

    public static CompoundName INPUT_ID = new CompoundName("id");
    public static CompoundName USE_ABSTRACT_EMBEDDING = new CompoundName("use-abstract");
    public static CompoundName REMOVE_ARTICLE_FROM_RESULT =  new CompoundName("remove-id");

    public static String TENSOR_SUMMARY = "attributeprefetch";

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
        boolean includeAbstract = query.properties().getBoolean(USE_ABSTRACT_EMBEDDING,false);
        boolean removeArticle = query.properties().getBoolean(REMOVE_ARTICLE_FROM_RESULT,true);
        Integer id = query.properties().getInteger(INPUT_ID, null);
        Integer traverseId = extractRelatedToItem(query.getModel().getQueryTree());
        if (id == null && traverseId == null) { //Just return whatever this query was about
            query.trace("Did not find any related_id in the query",3);
           return execution.search(query);
        }
        if(traverseId != null)  {
            id = traverseId;
        }
        Article article = getArticle(id,execution,query);
        Query relatedQuery = generateRelatedQuery(article,query,includeAbstract);


        if(removeArticle) {
            NotItem notItem = new NotItem();
            notItem.addPositiveItem(relatedQuery.getModel().getQueryTree().getRoot());
            notItem.addNegativeItem(new WordItem(id.toString(), "id", true));
            relatedQuery.getModel().getQueryTree().setRoot(notItem);
        }
        return execution.search(relatedQuery);
    }

    private Article getArticle(Integer id, Execution execution,Query query) {
        Query articleQuery = new Query();
        query.attachContext(articleQuery);
        articleQuery.getPresentation().setSummary(TENSOR_SUMMARY);
        WordItem idFilter = new WordItem(id.toString(), "id", true);
        articleQuery.getModel().getQueryTree().setRoot(idFilter);
        articleQuery.getModel().setRestrict("doc");
        articleQuery.setHits(1);
        articleQuery.getRanking().setProfile("unranked");
        Result articleResult = execution.search(articleQuery);
        execution.fill(articleResult,TENSOR_SUMMARY);
        return extractFromResult(articleResult);
    }

    /**
     * Finds a related_to item in the query and removes it
     *
     * @return the id specified by the related_to item, or null if none
     */
    private Integer extractRelatedToItem(CompositeItem parent) {
        for (Item child : parent.items()) {
            if (child instanceof IntItem) {
                IntItem intItem = (IntItem) child;
                if (intItem.getIndexName().equals(RELATED_TO_FIELD)) {
                    parent.removeItem(child);
                    return Integer.parseInt(intItem.getRawWord());
                }
            } else if (child instanceof CompositeItem) {
                Integer relatedId = extractRelatedToItem((CompositeItem) child);
                if (relatedId != null)
                    return relatedId;
            }
        }
        return null;
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
     * @return The related query
     */

    private Query generateRelatedQuery(Article a, Query originalQuery, boolean includeAbstract) {
        Query relatedQuery = originalQuery.clone();
        relatedQuery.getSelect().setGroupingExpressionString(originalQuery.getSelect().getGroupingExpressionString());
        Item root = relatedQuery.getModel().getQueryTree().getRoot();
        if(root instanceof IntItem) {
            IntItem r = (IntItem)root;
            if(r.getIndexName().equals(RELATED_TO_FIELD)) {
                root = new NullItem();
            }
        }
        Item nnRoot;
        NearestNeighborItem nnTitle = new NearestNeighborItem("title_embedding",
                                                              "title_vector");
        nnTitle.setAllowApproximate(false);
        nnTitle.setTargetNumHits(100);
        relatedQuery.getRanking().getFeatures().put("query(title_vector)", a.title);

        if(includeAbstract && a.article_abstract != null) {
            NearestNeighborItem nnAbstract = new NearestNeighborItem("abstract_embedding",
                    "abstract_vector");
            nnAbstract.setAllowApproximate(false);
            nnAbstract.setTargetNumHits(100);
            relatedQuery.getRanking().getFeatures().put("query(abstract_vector)", a.article_abstract);
            nnRoot = new OrItem();
            ((OrItem) nnRoot).addItem(nnAbstract);
            ((OrItem) nnRoot).addItem(nnTitle);
        } else  {
            nnRoot = nnTitle;
        }
        //Combine
        if ( ! hasTextTerms(root)) {
            relatedQuery.getModel().getQueryTree().setRoot(nnRoot);
            //query is empty must rank by vectors
            relatedQuery.getRanking().setProfile("related-ann");
        } else {
            AndItem andItem = new AndItem();
            andItem.addItem(root);
            andItem.addItem(nnRoot);
            relatedQuery.getModel().getQueryTree().setRoot(andItem);
        }
        return relatedQuery;
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

}
