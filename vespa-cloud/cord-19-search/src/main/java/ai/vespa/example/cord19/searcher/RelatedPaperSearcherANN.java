// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.cord19.searcher;




import com.google.inject.Inject;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.prelude.query.WordItem;
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
    private Linguistics linguistics;
    public static String SUMMARY = "attributeprefetch";

    @Inject
    public RelatedPaperSearcherANN(Linguistics linguistics) {
        this.linguistics = linguistics;
    }

    class Article {
        Tensor title;
        Tensor article_abstract;

        Article(Tensor title, Tensor article_abstract) {
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
        Article article = extractFromResult(result);
        if (article == null) {
            return new Result(query);
        }
        Query relatedQuery = generateRelatedQuery(article,includeAbstract);
        relatedQuery.getPresentation().setSummary("full");
        relatedQuery.getPresentation().setBolding(false);
        relatedQuery.setHits(query.getHits());
        query.attachContext(relatedQuery);
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

    private Query generateRelatedQuery(Article a, boolean includeAbstract) {
        Query relatedQuery = new Query();
        NearestNeighborItem nn = new NearestNeighborItem("title_embedding",
                "vector");
        nn.setAllowApproximate(false);
        nn.setTargetNumHits(10);
        relatedQuery.getRanking().getFeatures().put("query(vector)", a.title);
        relatedQuery.getRanking().setProfile(("semantic-search-title"));
        relatedQuery.getModel().getQueryTree().setRoot(nn);
        return relatedQuery;
    }

}
