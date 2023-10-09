// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.prelude.query.NearestNeighborItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;

public class TextEmbeddingSearcher extends Searcher {

    private final ModelsEvaluator modelsEvaluator;
    private final BPETokenizer tokenizer;

    public TextEmbeddingSearcher(ModelsEvaluator modelsEvaluator, BPETokenizer tokenizer) {
        this.modelsEvaluator = modelsEvaluator;
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        // Get input
        String inputString = query.properties().getString("input",null);
        if(inputString == null || inputString.isBlank())
            return new Result(query, ErrorMessage.createBadRequest("No 'input' query param"));

        // Tokenize input
        Tensor input = tokenizer.encode(inputString).rename("d0", "d1").expand("d0");

        // Evaluate transformer model to generate embedding
        Tensor embedding = modelsEvaluator.evaluatorOf("transformer").bind("input", input).evaluate();

        // Embedding tensor type is d0[1],d1[512]. Transform to expected x[512] type. And normalize.
        embedding = Util.slice(embedding, "d0:0").rename("d1", "x").l2Normalize("x");

        // Add this tensor to query
        query.getRanking().getFeatures().put("query(vit_b_32_text)", embedding);

        // Set up the nearest neighbor retrieval
        NearestNeighborItem nn = new NearestNeighborItem("vit_b_32_image", "vit_b_32_text");
        nn.setAllowApproximate(true);
        nn.setTargetNumHits(10);
        nn.setHnswExploreAdditionalHits(100);
        query.getModel().getQueryTree().setRoot(nn);

        // Set ranking profile
        query.getRanking().setProfile("vit_b_32_similarity");

        // Continue processing
        return execution.search(query);
    }

}
