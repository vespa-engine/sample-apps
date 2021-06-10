// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.searcher.colbert;

import ai.vespa.colbert.ColbertConfig;
import com.google.inject.Inject;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorHit;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import com.yahoo.tensor.TensorType;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.tensor.functions.Reduce;

import java.util.ArrayList;
import java.util.List;

/**
 * Searcher which prepares the input to the *query* encoder of ColBERT.
 * The evaluation of the trained BERT model is done at the content node(s) using onnx
 */
public class ColBERTSearcher extends Searcher {

    // Used by the backend onnx model
    private static final String inputIdsTensorName = "query(input_ids)";
    private static final String attentionMaskTensorName = "query(attention_mask)";

     // The tensor used when re-ranking passages
    private static final String colbertTensorName = "query(qt)";

    private final TensorType colbertTensorType;
    private final TensorType encoderTensorType;

    private final int query_max_length;
    private final int dim;
    private final String questionEncoderRankProfile;
    private final String outputName;
    private final BertTokenizer tokenizer;

    @Inject
    public ColBERTSearcher(BertTokenizer tokenizer, ColbertConfig config)  {
        this.tokenizer = tokenizer;
        this.query_max_length = config.max_query_length();
        this.dim = config.dim();
        this.questionEncoderRankProfile = config.rank_profile();
        this.outputName = config.output_name();
        this.encoderTensorType = TensorType.fromSpec("tensor<float>(d0[1],d1[" + this.dim + "])");
        this.colbertTensorType = TensorType.fromSpec("tensor<float>(qt{},x[" + this.dim + "])");
    }

    @Override
    public Result search(Query query, Execution execution) {
        Tensor colBertTensor =
                rewriteTensor(
                        getEmbedding(getQueryEmbeddingQuery(query),
                                     execution));
        Result result = new Result(query);
        result.hits().setSource("colbert");
        Hit tensorHit = new Hit("tensor");
        tensorHit.setSource("colbert");
        tensorHit.setField("tensor", colBertTensor);
        result.hits().add(tensorHit);
        return result;
    }

    protected Tensor getEmbedding(Query embeddingQuery, Execution execution) throws RuntimeException{
        Result r = execution.search(embeddingQuery);
        execution.fill(r);
        ErrorHit errorHit = r.hits().getErrorHit();
        if (errorHit != null)
            throw new RuntimeException(errorHit.toString());
        FeatureData featureData = (FeatureData)r.hits().get(0).getField("summaryfeatures");
        if (featureData == null)
            throw new RuntimeException("No summary produced but no error hit");
        return featureData.getTensor(this.outputName);
    }

    /**
     * Builds the input_ids tensor which for colbert is
     * [CLS] [unused0] query_token_ids [SEP] [MASK] [MASK]...]
     *
     * @param originalQuery the original query instance
     * @return a query to execute to get the query embeddings
     */
    protected Query getQueryEmbeddingQuery(Query originalQuery)  {
        String queryString = originalQuery.getModel().getQueryString();
        int CLS_TOKEN_ID = 101;  // [CLS]
        int SEP_TOKEN_ID = 102;  // [SEP]
        int MASK_TOKEN_ID = 103; // [MASK]
        int Q_TOKEN_ID = 1; // [unused0] token id used during training to represent the query. unused1 for document.

        List<Integer> token_ids = this.tokenizer.tokenize(queryString, query_max_length, false);
        List<Integer> input_ids = new ArrayList<>(query_max_length);
        List<Integer> attention_mask = new ArrayList<>(query_max_length);

        input_ids.add(CLS_TOKEN_ID);
        input_ids.add(Q_TOKEN_ID);
        input_ids.addAll(token_ids);
        input_ids.add(SEP_TOKEN_ID);
        int length = input_ids.size();
        // Pad up to max length with mask token_id
        int padding = query_max_length - length;
        for (int i = 0; i < padding; i++)
            input_ids.add(MASK_TOKEN_ID);

        for (int i = 0; i < length; i++)
            attention_mask.add(1);
        for (int i = 0; i < padding; i++)
            attention_mask.add(0);

        Query query = new Query();
        query.setTimeout("5s");
        originalQuery.attachContext(query);
        query.setHits(1);
        query.getRanking().setProfile(this.questionEncoderRankProfile);
        query.getModel().setRestrict("query");
        query.getModel().getQueryTree().setRoot(new WordItem("query", "sddocname"));
        query.getRanking().getFeatures().put(inputIdsTensorName, toTensor(input_ids));
        query.getRanking().getFeatures().put(attentionMaskTensorName, toTensor(attention_mask));
        return query;
    }

    /**
     * Removes the batch dimension ("d0") from the given tensor
     *
     * @param embedding the Tensor from query encoding with batch dimension
     */
    private Tensor rewriteTensor(Tensor embedding) {
        Tensor t = embedding.reduce(Reduce.Aggregator.min, "d0");
        Tensor.Builder builder = Tensor.Builder.of(colbertTensorType);
        for (int i = 0; i < query_max_length; i++)
            for (int j = 0; j< dim; j++)
                builder.cell(TensorAddress.of(i, j), t.get(TensorAddress.of(i, j)));
        return builder.build();
    }

    /**
     * Convert to tensor representation with batch dim
     *
     * @param input the List of token_ids
     * @return tensor representation with batch dim
     */
    private Tensor toTensor(List<Integer> input) {
        Tensor.Builder builder = Tensor.Builder.of(encoderTensorType);
        int i = 0;
        for (Integer in:input)  {
            if (i == dim) break;
            builder.cell(TensorAddress.of(0, i), in); // 0 is batch dim
            i++;
        }
        return builder.build();
    }

}
