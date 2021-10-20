// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

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

/**
 * This Searcher encodes the query using the sentence transformer model
 * https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3
 */

public class QueryEmbeddingSearcher extends Searcher {

    private static final Tensor BATCH_TENSOR = Tensor.from("tensor<float>(d0[1]):[1]");
    private final Model encoder;
    private static final int maxQueryLength = 32;

    @Inject
    public QueryEmbeddingSearcher(ModelsEvaluator evaluator) {
        this.encoder = evaluator.requireModel("dense_encoder");
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
        int CLS_TOKEN_ID = 101; // [CLS]
        int SEP_TOKEN_ID = 102; // [SEP]
        QueryTensorInput queryTensorInput = QueryTensorInput.getFrom(originalQuery.properties());
        List<Integer> queryTokenIds = queryTensorInput.getQueryTokenIds();
        if(queryTokenIds.size() > maxQueryLength -2)
            queryTokenIds.subList(0,maxQueryLength-2);
        List<Integer> inputIds = new ArrayList<>(queryTokenIds.size()+2 );
        inputIds.add(CLS_TOKEN_ID);
        inputIds.addAll(queryTokenIds);
        inputIds.add(SEP_TOKEN_ID);
        Tensor input_sequence = queryTensorInput.getTensorRepresentation(inputIds,"d1");
        Tensor attentionMask = createAttentionMask(input_sequence);
        return rewriteTensor(this.encoder.evaluatorOf().
                bind("input_ids",input_sequence.multiply(BATCH_TENSOR)).
                bind("attention_mask",attentionMask.multiply(BATCH_TENSOR)).evaluate());
    }

    protected static Tensor createAttentionMask(Tensor d)  {
        return d.map((x) -> x > 0 ? 1:0);
    }

    protected static Tensor rewriteTensor(Tensor embedding) {
        Tensor t = embedding.reduce(Reduce.Aggregator.min, "d0");
        return t.rename("d1","d0");
    }
}

