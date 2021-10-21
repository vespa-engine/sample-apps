// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.cord19.searcher;

import com.yahoo.prelude.query.CompositeItem;
import com.yahoo.prelude.query.IntItem;
import com.yahoo.prelude.query.Item;
import com.yahoo.prelude.query.NotItem;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.processing.request.CompoundName;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import java.util.Optional;

/**
 * Superclass of searchers finding related articles by various methods.
 *
 * @author jobergum
 */
abstract class RelatedArticlesSearcher extends Searcher {

    private static final String relatedToField = "related_to";

    private static CompoundName inputId = new CompoundName("id");
    private static CompoundName useAbstractEmbedding = new CompoundName("use-abstract");
    private static CompoundName removeArticleFromResult =  new CompoundName("remove-id");

    @Override
    public Result search(Query query, Execution execution) {
        boolean includeAbstract = query.properties().getBoolean(useAbstractEmbedding, true);
        boolean filterRelatedArticle = query.properties().getBoolean(removeArticleFromResult, true);

        Optional<Integer> relatedArticleId = relatedArticleIdFrom(query);
        if (relatedArticleId.isEmpty()) return execution.search(query); // Not a related search; Pass on

        addRelatedItem(relatedArticleId.get(), includeAbstract, execution, query);

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

    /** Add the query tree item (and any other decoration needed) to find related articles */
    protected abstract void addRelatedItem(Integer relatedArticleId,
                                           boolean includeAbstract,
                                           Execution execution,
                                           Query query);

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

    private void addArticleFilterTerm(Integer relatedArticleId, Query query) {
        NotItem notItem = new NotItem();
        notItem.addPositiveItem(query.getModel().getQueryTree().getRoot());
        notItem.addNegativeItem(new WordItem(relatedArticleId.toString(), "id", true));
        query.getModel().getQueryTree().setRoot(notItem);
    }

}
