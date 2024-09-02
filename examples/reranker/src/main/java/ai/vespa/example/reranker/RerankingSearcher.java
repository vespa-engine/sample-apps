// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.reranker;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.processing.request.CompoundName;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;

import java.util.Iterator;

/**
 * A searcher which can rerank results from another Vespa application.
 *
 * @author bratseth
 */
public class RerankingSearcher extends Searcher {

    public static final CompoundName rerankHitsParameter = new CompoundName("rerank.hits");
    public static final CompoundName rerankModelParameter = new CompoundName("rerank.model");

    private final ModelsEvaluator modelsEvaluator;

    private final int defaultRerankHits;
    private final String defaultRerankProfile;
    private final String defaultRerankModel;

    public RerankingSearcher(RerankerConfig config, ModelsEvaluator modelsEvaluator) {
        this.modelsEvaluator = modelsEvaluator;

        this.defaultRerankHits = config.rerank().hits();
        this.defaultRerankProfile = config.rerank().profile();
        this.defaultRerankModel = config.rerank().model();
    }

    @Override
    public Result search(Query query, Execution execution) {
        query.setHits(Math.max(query.getHits(), query.properties().getInteger(rerankHitsParameter, defaultRerankHits)));
        if (query.getRanking().getProfile().equals("default"))
            query.getRanking().setProfile(defaultRerankProfile);

        Result result = execution.search(query);
        rerank(result, query.properties().getString(rerankModelParameter, defaultRerankModel));
        return result;
    }

    private void rerank(Result result, String rerankModel) {
        for (Iterator<Hit> i = result.hits().unorderedDeepIterator(); i.hasNext(); ) {
            Hit hit = i.next();
            if ( ! hit.isAuxiliary())
                rerank(hit, rerankModel);
        }
    }

    private void rerank(Hit hit, String rerankModel) {
        FunctionEvaluator evaluator = modelsEvaluator.evaluatorOf(rerankModel);

        FeatureData features = (FeatureData)hit.getField("summaryfeatures");
        if (features == null)
            throw new IllegalArgumentException("Missing 'summaryfeatures' field in " + hit +
                                               ". Use a rank profile with a 'summary-features' block, using '" +
                                               hit.getQuery().getRanking().getProfile() + "'");
        for (String featureName : features.featureNames()) {
            if (featureName.equals("vespa.summaryFeatures.cached")) continue;
            if (evaluator.context().arguments().contains(featureName))
                evaluator.bind(featureName, features.getTensor(featureName));
        }
        Tensor result = evaluator.evaluate();
        hit.setRelevance(result.asDouble());
    }

}
