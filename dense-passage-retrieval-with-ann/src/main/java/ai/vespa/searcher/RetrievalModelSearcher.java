// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import ai.vespa.QuestionAnswering;
import ai.vespa.models.evaluation.FunctionEvaluator;
import ai.vespa.models.evaluation.ModelsEvaluator;
import ai.vespa.tokenizer.BertTokenizer;
import com.google.inject.Inject;
import com.yahoo.component.chain.dependencies.Before;
import com.yahoo.prelude.query.*;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;

import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.ConstantTensor;
import com.yahoo.tensor.functions.Slice;
import java.util.BitSet;
import java.util.List;


@Before("QuestionAnswering")
public class RetrievalModelSearcher extends Searcher {

    private final TensorType queryHashEmbeddingType = TensorType.fromSpec("tensor<int8>(d0[96])");
    private static String QUERY_EMBEDDING_TENSOR_NAME = "query(query_embedding)";
    private static String QUERY_HASH_TENSOR_NAME = "query(query_hash)";

    private final BertTokenizer tokenizer;

    private final ModelsEvaluator modelsEvaluator;

    @Inject
    public RetrievalModelSearcher(BertTokenizer tokenizer, ModelsEvaluator evaluator) {
        this.tokenizer = tokenizer;
        this.modelsEvaluator = evaluator;
    }

    @Override
    public Result search(Query query, Execution execution) {
        String queryInput = query.getModel().getQueryString();
        if(query.getModel().getQueryString() == null ||
                query.getModel().getQueryString().length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));

        switch (QuestionAnswering.getRetrivalMethod(query)) {
            case DENSE:
                Tensor clsEmbedding = getEmbeddingTensor(queryInput, query);
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
        nn.setTargetNumHits(1000); //Number we want to retrieve for re-ranking

        if(query.properties().getBoolean("ann.brute-force") )
            nn.setAllowApproximate(false);
        else
            nn.setAllowApproximate(true);

        nn.setHnswExploreAdditionalHits(query.properties().getInteger("ann.extra-hits",0));
        query.getRanking().getFeatures().put(QUERY_EMBEDDING_TENSOR_NAME , questionEmbedding);
        query.getRanking().getFeatures().put(QUERY_HASH_TENSOR_NAME,hashEmbedding);
        return nn;
    }

    /**
     * Fetches the DPR query embedding vector
     * @param queryInput The text representation of the query
     * @param query The original query
     * @return The embedding tensor as produced by the DPR query encoder
     */
    protected Tensor getEmbeddingTensor(String queryInput, Query query) {
        long start = System.currentTimeMillis();

        List<Integer> tokenIds = tokenizer.tokenize(queryInput, 32);

        FunctionEvaluator evaluator = modelsEvaluator.evaluatorOf("question_encoder", "output_0");
        evaluator.bind("input_ids", transformTokenInputIds(tokenIds));
        evaluator.bind("token_type_ids", transformTokenTypeIds(tokenIds));
        evaluator.bind("attention_mask", transformTokenAttentionMask(tokenIds));

        Tensor embedding = evaluator.evaluate();
        Tensor cls = slice(embedding, "d0", 0, "d1", 0).rename("d2", "x");  // Slice out CLS embedding

        long duration = System.currentTimeMillis() - start;
        query.trace("Encoder phase took " + duration + "ms" , 3);

        return cls;
    }

    /**
     * Rewrites from float vector to a int8 tensor
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
        Tensor.Builder builder =  Tensor.Builder.of(queryHashEmbeddingType);
        for(int i = 0; i < 96; i++)
            builder.cell().label("d0", String.valueOf(i)).value(bytes[96 - i -1]);
        return builder.build();
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
        for (int i = 0; i < tokens.size(); ++i) {
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

    private static Tensor slice (Tensor tensor, String d0, int i0, String d1, int i1) {
        return new Slice<>(new ConstantTensor<>(tensor),
                List.of(new Slice.DimensionValue<>(d0, i0),new Slice.DimensionValue<>(d1, i1)))
                .evaluate();
    }

}
