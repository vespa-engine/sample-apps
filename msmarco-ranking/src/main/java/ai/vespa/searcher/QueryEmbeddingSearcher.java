// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import ai.vespa.embedding.DenseEmbedder;
import ai.vespa.models.evaluation.Model;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.google.inject.Inject;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.functions.Reduce;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * This Searcher encodes the query using the sentence transformer model
 * https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3
 */

public class QueryEmbeddingSearcher extends Searcher {

    private static final Tensor BATCH_TENSOR = Tensor.from("tensor<float>(d0[1]):[1]");
    private final DenseEmbedder embedder;
    private int maxLength = 32;

    @Inject
    public QueryEmbeddingSearcher(DenseEmbedder embedder) {
        this.embedder = embedder;
    }

    @Override
    public Result search(Query query, Execution execution) {
        Tensor embedding =  getEmbedding(query);
        Result result = new Result(query);
        result.hits().setSource("embedding");
        Hit tensorHit = new Hit("tensor");
        tensorHit.setSource("embedding");
        tensorHit.setField("tensor", embedding);
        result.hits().add(tensorHit);
        result.setTotalHitCount(1);
        return result;
    }

    protected Tensor getEmbedding(Query originalQuery)  {
        QueryTensorInput queryTensorInput = QueryTensorInput.getFrom(originalQuery.properties());
        List<Integer> queryTokenIds = queryTensorInput.getQueryTokenIds();
        Optional<Tensor> result = embedder.embed(queryTokenIds,maxLength);
        if(result.isEmpty())
            return null;
        return result.get();
    }
}

