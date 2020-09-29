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
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.Reduce;


import java.util.List;


@Before("QuestionAnswering")
public class RetrieveModelSearcher extends Searcher {

    private static String QUERY_TENSOR_NAME = "query(query_token_ids)";
    TensorType questionInputTensorType = TensorType.fromSpec("tensor<float>(d0[32])");
    private static String QUERY_EMBEDDING_TENSOR_NAME = "query(query_embedding)";

    BertTokenizer tokenizer;

    @Inject
    public RetrieveModelSearcher(BertTokenizer tokenizer) {
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
                query.getModel().getQueryTree().setRoot(denseRetrieval(getEmbeddingTensor(questionTokenIds, execution, query), query));
                query.getRanking().setProfile("dense");
                break;
            case SPARSE:
                query.getModel().getQueryTree().setRoot(sparseRetrieval(queryInput, query));
                query.getRanking().setProfile("sparse");
                break;
            case HYBRID:
                Item ann = denseRetrieval(getEmbeddingTensor(questionTokenIds, execution, query),query);
                Item wand = sparseRetrieval(queryInput,query);
                OrItem disjunction = new OrItem();
                disjunction.addItem(ann);
                disjunction.addItem(wand);
                query.getModel().getQueryTree().setRoot(disjunction);
                query.getRanking().setProfile("hybrid");
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

    private Item denseRetrieval(Tensor questionEmbedding, Query query) {
        NearestNeighborItem nn = new NearestNeighborItem("text_embedding", "query_embedding");
        nn.setTargetNumHits(query.getHits());
        nn.setAllowApproximate(true);
        nn.setHnswExploreAdditionalHits(query.properties().getInteger("ann.extra",1000));
        Tensor l2convertedTensor = rewriteQueryTensor(questionEmbedding);
        query.getRanking().getFeatures().put(QUERY_EMBEDDING_TENSOR_NAME , l2convertedTensor);
        return nn;
    }

    private Tensor getEmbeddingTensor(Tensor questionTensor, Execution execution, Query query) {
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
        long duration = System.currentTimeMillis() - start;
        embeddingModelQuery.trace("Encoder phase took " + duration + "ms" , 3);
        FeatureData featureData = (FeatureData)embeddingResult.hits().get(0).getField("summaryfeatures");
        return featureData.getTensor("onnxModel(encoder).embedding");
    }

    private Tensor getQueryTokenIds(String queryInput, int maxLength) {
        List<Integer> tokensIds = tokenizer.tokenize(queryInput, maxLength,true);
        Tensor.Builder builder = Tensor.Builder.of(questionInputTensorType);
        int i = 0;
        for(Integer tokenId: tokensIds)
            builder.cell(tokenId,i++);
        return builder.build();
    }

    /**
     * Rewrites the question embedding tensor and remove the batch dimension, grow from 768 to 769
     * @param embedding the question embedding returned from query encoder
     * @return a rewritten tensor.
     */
    private Tensor rewriteQueryTensor(Tensor embedding) {
        return embedding.reduce(Reduce.Aggregator.min, "d0").concat(0, "d1").rename("d1", "x");
    }


}
