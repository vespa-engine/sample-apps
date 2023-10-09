// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.searcher;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.dependencies.After;
import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.HitGroup;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.TensorType;
import java.util.Set;

@After("ExternalYql")
@Before("Ranking")
public class DeDupingSearcher extends Searcher {
    private final ModelsEvaluator modelsEvaluator;

    public DeDupingSearcher(ModelsEvaluator evaluator) {
        this.modelsEvaluator = evaluator;
    }

    @Override
    public Result search(Query query, Execution execution) {
        Set<String> summaryFields = query.getPresentation().getSummaryFields();
        final boolean userWantsVector = summaryFields.isEmpty() || summaryFields.contains("vector");
        if (!userWantsVector) //User request does not ask for vector - but we need it.
            query.getPresentation().getSummaryFields().add("vector");

        int userHits = query.getHits();
        Result result = execution.search(query);

        if (query.properties().getBoolean("collapse.enable", true)){
            ensureFilled(result, "vector-summary", execution);
            result = dedup(result, userHits);
        }
        result.hits().trim(0,userHits);
        if (!userWantsVector) { //User did not ask for it - remove it.
            result.hits().forEach(h -> h.removeField("vector"));
            query.getPresentation().getSummaryFields().remove("vector");
        }
        return result;
    }

    /**
     * Deduping based on vector similarity
     * @param result the result to dedupe
     * @param userHits the number of hits requested by the user
     * @return
     */

    public Result dedup(Result result, int userHits){
        if(result.getTotalHitCount() == 0 || result.hits().getError() != null)
            return result;

        double similarityThreshold = result.getQuery().properties().
                getDouble("collapse.similarity.threshold", 0.95);

        int maxHits = result.getQuery().properties().
                getInteger("collapse.similarity.max-hits",1000);

        int size = Math.min(result.getHitCount(), maxHits);
        //Iterate over the diagonal and for
        //each hit see if we already added
        //a hit with high similarity to the current image i
        IndexedTensor similarityMatrix = getSimilarityMatrix(result, size);
        HitGroup uniqueHits = new HitGroup();
        for (int i = 0; i < size; i++) {
            double maxSim = 0;
            for(int j = i -1; j >= 0; j--)  {
                float sim = similarityMatrix.getFloat(i,j);
                if(sim > maxSim)
                    maxSim = sim;
            }
            if (maxSim < similarityThreshold) {
                uniqueHits.add(result.hits().get(i));
                if(uniqueHits.size() == userHits)
                    break;
            }
        }
        result.setHits(uniqueHits);
        return result;
    }

    public IndexedTensor getSimilarityMatrix(Result result, int size) {
        TensorType type = new TensorType.Builder(TensorType.Value.FLOAT).
                indexed("d0", size).indexed("d1", 768).build();
        IndexedTensor.Builder builder = IndexedTensor.Builder.of(type);
        HitGroup hits = result.hits();
        for (int i = 0; i < size; i++) {
            IndexedTensor vector = (IndexedTensor) hits.get(i).getField("vector");
            for (int j = 0; j < vector.size(); j++)
                builder.cell(vector.get(j), i, j);
        }
        // Perform N X N similarity
        FunctionEvaluator similarity = modelsEvaluator.
                evaluatorOf("vespa_pairwise_similarity");
       return (IndexedTensor) similarity.bind(
                "documents", builder.build()).evaluate();
    }
}
