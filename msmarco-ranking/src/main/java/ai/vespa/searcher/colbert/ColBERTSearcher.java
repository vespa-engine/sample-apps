package ai.vespa.searcher.colbert;

import ai.vespa.colbert.ColbertConfig;
import com.google.inject.Inject;
import com.yahoo.prelude.query.WordItem;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorHit;
import com.yahoo.search.result.FeatureData;
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
 * The evaluation of the trained BERT model is done at the content node
 */

public class ColBERTSearcher extends Searcher {

    /**
     * Named tensors used when evaluating the Colbert embeddings
     */
    private static String inputIdsTensorName = "query(input_ids)";
    private static String attentionMaskTensorName = "query(attention_mask)";

    /**
     * The tensor from used when ranking documents/passages
     */

    private static String colbertTensorName = "query(qt)";

    private TensorType colbertTensorType;
    private TensorType encoderTensorType;

    private int query_max_length;
    private int dim;
    private String questionEncoderRankProfile;
    private String outputName;
    private BertTokenizer tokenizer;


    @Inject
    public ColBERTSearcher(BertTokenizer tokenizer, ColbertConfig config)  {
        this.tokenizer = tokenizer;
        this.query_max_length = config.max_query_length();
        this.dim = config.dim();
        this.questionEncoderRankProfile = config.rank_profile();
        this.outputName = config.output_name();
        this.encoderTensorType = TensorType.fromSpec("tensor<float>(d0[1],t[" + this.dim + "])");
        this.colbertTensorType = TensorType.fromSpec("tensor<float>(qt{},x[" + this.dim + "])");
    }

    @Override
    public Result search(Query query, Execution execution) {
        query.getRanking().getFeatures().put(colbertTensorName,
                rewriteTensor(
                        getEmbedding(getQueryEmbeddingQuery(query),
                                execution)));
        return execution.search(query);
    }

    protected Tensor getEmbedding(Query embeddingQuery, Execution execution) throws RuntimeException{
        Result r = execution.search(embeddingQuery);
        execution.fill(r);
        ErrorHit errorHit = r.hits().getErrorHit();
        if(errorHit != null) {
            throw new RuntimeException(errorHit.toString());
        }
        FeatureData featureData = (FeatureData)r.hits().get(0).getField("summaryfeatures");
        if(featureData == null)
            throw new RuntimeException("No summary produced but no error hit");
        return featureData.getTensor(this.outputName);
    }


    /**
     *Builds the input_ids tensor which for colbert is
     * [CLS] [unused0] query_token_ids [SEP] [MASK] [MASK]...]
     *
     * @param originalQuery The original query instance
     * @return a query to execute to get the query embeddings
     */

    protected Query getQueryEmbeddingQuery(Query originalQuery)  {
        String queryString = originalQuery.getModel().getQueryString();
        int CLS_TOKEN_ID = 101; //[CLS]
        int SEP_TOKEN_ID = 102; //[SEP]
        int MASK_TOKEN_ID = 103;// [MASK]
        int Q_TOKEN_ID = 1; //[unused0] token id used during training to represent Query

        List<Integer> token_ids = this.tokenizer.tokenize(queryString,query_max_length,false);
        List<Integer> input_ids = new ArrayList<>(query_max_length);
        List<Integer> attention_mask = new ArrayList<>(query_max_length);

        input_ids.add(CLS_TOKEN_ID);
        input_ids.add(Q_TOKEN_ID);
        input_ids.addAll(token_ids);
        input_ids.add(SEP_TOKEN_ID);
        int length = input_ids.size();
        //Pad up to max length with mask token_id
        int padding = query_max_length - length;
        for(int i = 0; i < padding; i++)
            input_ids.add(MASK_TOKEN_ID);

        for(int i = 0; i < length;i++)
            attention_mask.add(1);
        for(int i = 0; i < padding;i++)
            attention_mask.add(0);

        Query query = new Query();
        originalQuery.attachContext(query);
        query.setHits(1);
        query.getRanking().setProfile(this.questionEncoderRankProfile);
        query.getModel().setRestrict("query");
        query.getModel().getQueryTree().setRoot(new WordItem("query","sddocname"));
        query.getRanking().getFeatures().put(inputIdsTensorName,toTensor(input_ids));
        query.getRanking().getFeatures().put(attentionMaskTensorName,toTensor(attention_mask));
        return query;
    }


    private Tensor rewriteTensor(Tensor embedding) {
        //remove batch dimension d0
        Tensor t = embedding.reduce(Reduce.Aggregator.min,"d0");
        Tensor.Builder builder = Tensor.Builder.of(colbertTensorType);
        for(int i = 0; i < query_max_length;i++)
            for(int j = 0; j< dim; j++)
                builder.cell(TensorAddress.of(i,j),t.get(TensorAddress.of(i,j)));
        return builder.build();
    }

    /**
     * Convert to tensor representation with batch dim
     * @param input the List of token_ids
     * @return tensor representation with batch dim
     */

    private Tensor toTensor(List<Integer> input) {
        Tensor.Builder builder = Tensor.Builder.of(encoderTensorType);
        int i = 0;
        for(Integer in:input)  {
            if (i == dim)
                break;
            builder.cell(TensorAddress.of(0,i),in);
            i++;
        }
        return builder.build();
    }

}
