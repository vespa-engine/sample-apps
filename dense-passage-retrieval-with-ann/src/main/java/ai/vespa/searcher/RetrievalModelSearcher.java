// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import ai.vespa.QuestionAnswering;
import ai.vespa.tokenizer.BertTokenizer;
import com.google.inject.Inject;
import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.prelude.query.*;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.Reduce;

import java.util.Arrays;
import java.util.BitSet;
import java.util.List;


@Before("QuestionAnswering")
public class RetrievalModelSearcher extends Searcher {

    private static String QUERY_TENSOR_NAME = "query(query_token_ids)";
    TensorType questionInputTensorType = TensorType.fromSpec("tensor<float>(d0[32])");
    TensorType queryHashEmbedding = TensorType.fromSpec("tensor<int8>(d0[96])");
    private static String QUERY_EMBEDDING_TENSOR_NAME = "query(query_embedding)";
    private static String QUERY_HASH_TENSOR_NAME = "query(query_hash)";

    BertTokenizer tokenizer;

    @Inject
    public RetrievalModelSearcher(BertTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        String queryInput = query.getModel().getQueryString();
        if(query.getModel().getQueryString() == null ||
                query.getModel().getQueryString().length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));

        Tensor questionTokenIds = getQueryTokenIds(queryInput, questionInputTensorType.sizeOfDimension("d0").get().intValue());
        query.getRanking().getFeatures().put(QUERY_TENSOR_NAME, questionTokenIds);

        switch (QuestionAnswering.getRetrivalMethod(query)) {
            case DENSE:
                Tensor floatQueryEmbedding = getEmbeddingTensor(questionTokenIds, execution, query);
                Tensor clsEmbedding = floatQueryEmbedding.rename("d2", "x");
                Tensor hashEmbedding = rewriteQueryTensor(clsEmbedding);
                query.getModel().getQueryTree().setRoot(denseRetrieval(clsEmbedding,hashEmbedding,query));
                query.getRanking().setProfile("dense");
                break;
            case SPARSE:
                query.getModel().getQueryTree().setRoot(sparseRetrieval(queryInput, query));
                query.getRanking().setProfile("sparse");
                break;
        }

        if(QuestionAnswering.isRetrieveOnly(query))
            query.getRanking().setProfile(query.getRanking().getProfile() + "-retriever");

        query.getModel().setRestrict("wiki");
        return execution.search(query);
    }

    private Item sparseRetrieval(String queryInput, Query query) {
        String[] tokens = queryInput.split(" ");
        WeakAndItem wand = new WeakAndItem();
        wand.setN(query.getHits());
        for(String t: tokens)
            wand.addItem(new WordItem(t,"default", true));
        return wand;
    }

    private Item denseRetrieval(Tensor questionEmbedding, Tensor hashEmbedding, Query query) {
        NearestNeighborItem nn = new NearestNeighborItem("hash", "query_hash");
        nn.setTargetNumHits(1000);
        nn.setAllowApproximate(true);
        nn.setHnswExploreAdditionalHits(query.properties().getInteger("ann.extra",0));
        query.getRanking().getFeatures().put(QUERY_EMBEDDING_TENSOR_NAME , questionEmbedding);
        query.getRanking().getFeatures().put(QUERY_HASH_TENSOR_NAME,hashEmbedding);
        return nn;
    }

    /**
     * Fetches the DPR query embedding vector
     * @param questionTensor The input tensor with the question token_ids
     * @param execution The execution to pass the new query
     * @param query The original query
     * @return The embedding tensor as produced by the DPR query encoder
     */

    protected Tensor getEmbeddingTensor(Tensor questionTensor, Execution execution, Query query) {
        long start = System.currentTimeMillis();
        Query embeddingModelQuery = new Query();
        query.attachContext(embeddingModelQuery);
        embeddingModelQuery.setHits(1);
        embeddingModelQuery.setTimeout(query.getTimeLeft());
        embeddingModelQuery.getRanking().setQueryCache(true);
        embeddingModelQuery.getModel().setRestrict("query");
        embeddingModelQuery.getRanking().setProfile("question_encoder");
        embeddingModelQuery.getModel().getQueryTree().setRoot(new WordItem("query","sddocname"));
        embeddingModelQuery.getRanking().getFeatures().put(QUERY_TENSOR_NAME, questionTensor);
        Result embeddingResult = execution.search(embeddingModelQuery);
        execution.fill(embeddingResult);
        if(embeddingResult.getTotalHitCount() == 0)
            throw new RuntimeException("No results for query document - Did you index the query document? ");
        long duration = System.currentTimeMillis() - start;
        embeddingModelQuery.trace("Encoder phase took " + duration + "ms" , 3);
        FeatureData featureData = (FeatureData)embeddingResult.hits().get(0).getField("summaryfeatures");
        return featureData.getTensor("rankingExpression(cls_embedding)");
    }

    /**
     * Encode the input question
     * @param queryInput The input question
     * @param maxLength The maximum sequence length reserved for the question
     * @return A tensor of type questionInputTensorType storing the token_ids
     */

    protected Tensor getQueryTokenIds(String queryInput, int maxLength) {
        List<Integer> tokensIds = tokenizer.tokenize(queryInput, maxLength,true);
        Tensor.Builder builder = Tensor.Builder.of(questionInputTensorType);
        int i = 0;
        for(Integer tokenId: tokensIds)
            builder.cell(tokenId,i++);
        return builder.build();
    }

    /**
     * Rewrites the question embedding tensor and remove the batch dimension, grow from 768 to 769
     *
     * @param embedding the question embedding returned from query encoder
     * @return a rewritten tensor.
     */
    protected Tensor rewriteQueryTensor(Tensor embedding) {
        BitSet set = new BitSet(768);
        for(int i = 0; i < 768;i++)  {
            if (embedding.get(TensorAddress.of(i)) > 0)
                set.set(768 - i -1);
        }
        byte[] bytes = set.toByteArray();
        Tensor.Builder builder =  Tensor.Builder.of(queryHashEmbedding);
        for(int i = 0; i < 96; i++)
            builder.cell().label("d0", String.valueOf(i)).value(bytes[96 - i -1]);
        return builder.build();
    }
}
