// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher;

import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.Model;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.IndexedTensor;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

public class ReRankingSearcher extends Searcher {

    private final Model model;
    private static final String MODEL_NAME = "msmarco_v2";
    private static final String TENSOR_TOKEN_FIELD_NAME = "text_token_ids";


    protected static class BertModelBatchInput  {
        IndexedTensor inputIds;
        IndexedTensor attentionMask;
        IndexedTensor tokenTypeIds;
        List<Hit> hits;

        BertModelBatchInput(IndexedTensor inputIds, IndexedTensor attentionMask, IndexedTensor tokenTypeIds,
                       List<Hit> hits)  {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
            this.tokenTypeIds = tokenTypeIds;
            this.hits = hits;
        }
    }

    public ReRankingSearcher(ModelsEvaluator modelsEvaluator) {
        this.model = modelsEvaluator.requireModel(MODEL_NAME);
    }

    @Override
    public Result search(Query query, Execution execution) {
        int hits = query.getHits();
        int reRankCount = query.getRanking().getRerankCount();
        query.setHits(reRankCount);
        query.getPresentation().getSummaryFields().add(TENSOR_TOKEN_FIELD_NAME);
        Result result = execution.search(query);
        execution.fill(result, TENSOR_TOKEN_FIELD_NAME);
        Result reRanked = reRank(result);
        reRanked.hits().trim(0,hits);
        for(Hit h:reRanked.hits())
            h.removeField(TENSOR_TOKEN_FIELD_NAME);
        return reRanked;
    }

    private Result reRank(Result result) {
        if(result.getConcreteHitCount() == 0)
            return result;
        List<Integer> queryTokens = QueryTensorInput.getFrom(result.getQuery().
                properties()).getQueryTokenIds();
        int maxSequenceLength = result.getQuery().
                properties().getInteger("rerank.sequence-length", 128);

        long start = System.currentTimeMillis();
        BertModelBatchInput input = buildModelInput(queryTokens,
                result,maxSequenceLength,TENSOR_TOKEN_FIELD_NAME);
        if(result.getQuery().getTrace().isTraceable(1))
            result.getQuery().trace("Prepare batch input took "
                    + (System.currentTimeMillis() - start)  + " ms",1);

        start = System.currentTimeMillis();
        batchInference(input);
        if(result.getQuery().getTrace().isTraceable(1))
            result.getQuery().trace("Inference batch took "
                    + (System.currentTimeMillis() - start)  + " ms",1);
        result.hits().sort();
        return result;
    }

    protected static List<Integer> trimToList(Tensor t) {
        int size = (int)t.size();
        List<Integer> tokens = new ArrayList<>(size);
        for(int i = 0; i < size; i++) {
            double value = t.get(TensorAddress.of(i));
            if(value > 0)
                tokens.add((int)value);
        }
        return tokens;
    }


    protected static BertModelBatchInput buildModelInput(List<Integer> queryTokens,
                                                         Result result,
                                                         int maxSequenceLength,
                                                         String tensorField) {
        if(maxSequenceLength < 3)
            maxSequenceLength = 3;

        List<List<Integer>> batch = new ArrayList<>(result.getHitCount());
        int maxPassageLength = 0;
        for (Hit h: result.hits()) {
            Tensor text = (Tensor) h.getField(tensorField);
            List<Integer> textTokens = trimToList(text);
            batch.add(textTokens);
            if (textTokens.size() > maxPassageLength)
                maxPassageLength = textTokens.size();
        }

        int sequenceLength = maxPassageLength + queryTokens.size() + 3;
        if(sequenceLength > maxSequenceLength)
            sequenceLength = maxSequenceLength;

        TensorType batchType = new TensorType.Builder(TensorType.Value.FLOAT).
                indexed("d0", result.hits().size()).indexed("d1",sequenceLength).build();
        IndexedTensor.Builder inputIdsBatchBuilder = IndexedTensor.Builder.of(batchType);
        IndexedTensor.Builder attentionMaskBatchBuilder = IndexedTensor.Builder.of(batchType);
        IndexedTensor.Builder tokenTypeIdsBatchBuilder = IndexedTensor.Builder.of(batchType);
        int queryLength = queryTokens.size();
        int batchId = 0;
        for (List<Integer> passage : batch) {
            int[] inputIds = new int[sequenceLength];
            byte[] attentionMask = new byte[sequenceLength];
            byte[] tokenType = new byte[sequenceLength];

            inputIds[0] = 101;
            attentionMask[0] = 1;
            tokenType[0] = 0;
            int index = 1;

            for (int j = 0; j < queryLength;j++) {
                if(index == sequenceLength -2 )
                    break;
                inputIds[index] = queryTokens.get(j);
                attentionMask[index] = 1;
                tokenType[index] = 0;
                index++;
            }
            inputIds[index] = 102;
            attentionMask[index] = 1;
            tokenType[index] = 0;
            index++;
            for (int j = 0; j < passage.size();j++) {
                if(index == sequenceLength -1)
                    break;
                inputIds[index] = passage.get(j);
                attentionMask[index] = 1;
                tokenType[index] = 1;
                index++;
            }
            inputIds[index] = 102;
            attentionMask[index] = 1;
            tokenType[index] = 1;

            for (int k = 0; k < sequenceLength; k++) {
                inputIdsBatchBuilder.cell(inputIds[k], batchId, k);
                attentionMaskBatchBuilder.cell(attentionMask[k], batchId, k);
                tokenTypeIdsBatchBuilder.cell(tokenType[k], batchId, k);
            }
            batchId++;
        }
        return new BertModelBatchInput(inputIdsBatchBuilder.build(),
                attentionMaskBatchBuilder.build(),
                tokenTypeIdsBatchBuilder.build(),
                result.hits().asList());
    }

    protected void batchInference(BertModelBatchInput input) {
        FunctionEvaluator evaluator = this.model.evaluatorOf();
        Tensor scores = evaluator.bind("input_ids",input.inputIds)
                .bind("attention_mask", input.attentionMask).
                bind("token_type_ids",input.tokenTypeIds).evaluate();
        ListIterator<Hit> it = input.hits.listIterator();
        while(it.hasNext()) {
            int index = it.nextIndex();
            Hit h = it.next();
            h.setRelevance(scores.get(TensorAddress.of(index,0)));
        }
    }
}
