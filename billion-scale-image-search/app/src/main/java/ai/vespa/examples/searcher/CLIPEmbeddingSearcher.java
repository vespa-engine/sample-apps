// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.examples.searcher;

import ai.vespa.examples.TensorUtils;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import ai.vespa.examples.BPETokenizer;

@Provides("ClipEmbedding")
public class CLIPEmbeddingSearcher extends Searcher {

    private final ModelsEvaluator modelsEvaluator;
    private final BPETokenizer tokenizer;
    private final String queryTensorName = "query(q)";
    private final String textInput ="prompt";

    public CLIPEmbeddingSearcher(ModelsEvaluator modelsEvaluator, BPETokenizer tokenizer) {
        this.modelsEvaluator = modelsEvaluator;
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        if(query.getRanking().getFeatures().getTensor(queryTensorName).isPresent())
            return execution.search(query);

        String inputString = query.properties().getString(textInput,null);
        if(inputString == null || inputString.isBlank())
            return execution.search(query);

        Tensor input = tokenizer.encode(inputString,"d1").expand("d0");
        // Evaluate transformer model to generate embedding
        Tensor embedding = modelsEvaluator.evaluatorOf("text_transformer").bind("input", input).evaluate();
        // Embedding tensor type is d0[1],d1[768]. Transform to expected x[768] type. And normalize.
        embedding = TensorUtils.slice(embedding, "d0:0").rename("d1", "x").l2Normalize("x");
        // Add this tensor to query
        query.getRanking().getFeatures().put(queryTensorName, embedding);
        return execution.search(query);
    }


}
