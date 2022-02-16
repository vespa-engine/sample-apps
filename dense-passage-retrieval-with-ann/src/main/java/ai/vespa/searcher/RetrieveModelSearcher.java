// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import ai.vespa.QuestionAnswering;
import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import com.google.inject.Inject;
import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.language.process.Embedder;
import com.yahoo.language.wordpiece.WordPieceEmbedder;
import com.yahoo.prelude.query.*;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import java.util.List;
import com.yahoo.tensor.functions.Reduce;


@Before("QuestionAnswering")
public class RetrieveModelSearcher extends Searcher {

    private final String QUERY_TOKEN_IDS_NAME = "query(query_token_ids)";
    private final String QUERY_EMBEDDING_TENSOR_NAME = "query(query_embedding)";

    private final TensorType queryTokenType = TensorType.fromSpec("tensor<float>(d0[32])");

    private final WordPieceEmbedder embedder;
    private final ModelsEvaluator modelsEvaluator;

    @Inject
    public RetrieveModelSearcher(WordPieceEmbedder embedder, ModelsEvaluator evaluator) {
        this.modelsEvaluator = evaluator;
        this.embedder = embedder;
    }

    @Override
    public Result search(Query query, Execution execution) {
        String queryInput = query.getModel().getQueryString();
        if(query.getModel().getQueryString() == null ||
                query.getModel().getQueryString().length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));

        Tensor.Builder builder = Tensor.Builder.of(queryTokenType);
        List<Integer> bertTokenIds = this.embedder.embed(queryInput, new Embedder.Context("q"));
        if(bertTokenIds.size() > 32)
            bertTokenIds = bertTokenIds.subList(0,32);
        int index = 0;
        for(Integer token : bertTokenIds) {
            builder.cell(TensorAddress.of(index), token);
            index++;
        }
        query.getRanking().getFeatures().put(QUERY_TOKEN_IDS_NAME,builder.build());

        switch (QuestionAnswering.getRetrivalMethod(query)) {
            case DENSE:
                query.getModel().getQueryTree().setRoot(denseRetrieval(getEmbeddingTensor(bertTokenIds), query));
                query.getRanking().setProfile("dense");
                break;
            case SPARSE:
                query.getModel().getQueryTree().setRoot(sparseRetrieval(queryInput, query));
                query.getRanking().setProfile("sparse");
                break;
            case HYBRID:
                Item ann = denseRetrieval(getEmbeddingTensor(bertTokenIds),query);
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

    private Item denseRetrieval(Tensor queryEmbedding, Query query) {
        NearestNeighborItem nn = new NearestNeighborItem("text_embedding", "query_embedding");
        nn.setTargetNumHits(query.getHits());
        nn.setAllowApproximate(true);
        nn.setHnswExploreAdditionalHits(query.properties().getInteger("ann.extra",10));
        query.getRanking().getFeatures().put(QUERY_EMBEDDING_TENSOR_NAME, queryEmbedding);
        return nn;
    }

    /**
     * Produce the DPR query embedding

     * @return The embedding tensor as produced by the DPR query encoder
     */

    private Tensor getEmbeddingTensor(List<Integer> tokenIds ) {
        FunctionEvaluator evaluator = modelsEvaluator.evaluatorOf("question_encoder", "output_0");
        evaluator.bind("input_ids", transformTokenInputIds(tokenIds));
        evaluator.bind("token_type_ids", transformTokenTypeIds(tokenIds));
        evaluator.bind("attention_mask", transformTokenAttentionMask(tokenIds));
        Tensor embedding = evaluator.evaluate();
        embedding = embedding.reduce(Reduce.Aggregator.min,"d0");
        embedding = embedding.rename("d1","x").concat(0,"x");
        return embedding;
    }



    private static Tensor.Builder tokenTensorBuilder(List<Integer> tokens) {
        TensorType.Dimension d0 = TensorType.Dimension.indexed("d0", 1);
        TensorType.Dimension d1 = TensorType.Dimension.indexed("d1", tokens.size() + 2);
        TensorType type = new TensorType(TensorType.Value.FLOAT, List.of(d0, d1));
        return Tensor.Builder.of(type);
    }

    private Tensor transformTokenInputIds(List<Integer> tokens) {
        Tensor.Builder builder = tokenTensorBuilder(tokens);
        builder.cell(101, 0);  // Starting CLS
        for (int i = 0; i < tokens.size(); i++) {
            builder.cell(tokens.get(i), i+1);
        }
        builder.cell(102, tokens.size() + 1);  // Ending SEP
        return builder.build();
    }

    private Tensor transformTokenTypeIds(List<Integer> tokens) {
        Tensor.Builder builder = tokenTensorBuilder(tokens);
        for (int i = 0; i < tokens.size() + 2; ++i) {
            builder.cell(0, i);
        }
        return builder.build();
    }

    private Tensor transformTokenAttentionMask(List<Integer> tokens) {
        Tensor.Builder builder = tokenTensorBuilder(tokens);
        for (int i = 0; i < tokens.size() + 2; ++i) {
            builder.cell(1, i);
        }
        return builder.build();
    }
}
