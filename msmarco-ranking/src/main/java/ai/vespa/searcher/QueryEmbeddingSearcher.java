package ai.vespa.searcher;


import ai.vespa.tokenizer.BertTokenizer;
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
import java.util.ArrayList;
import java.util.List;


/**
 * This class encodes the query using the sentence transformer model
 * https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3
 *
 * It builds the input tensor data
 *  - query(input_ids)
 *  - query(attention_mask)
 */

public class QueryEmbeddingSearcher extends Searcher {


    private static String inputIdsTensorName = "query(input_ids)";
    private static String attentionMaskTensorName = "query(attention_mask)";
    private TensorType inputType = TensorType.fromSpec("tensor<float>(d0[1],d1[32])");

    private BertTokenizer tokenizer;

    @Inject
    public QueryEmbeddingSearcher(BertTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        Tensor embedding =  getEmbedding(getQueryEmbeddingQuery(query),execution);
        Result result = new Result(query);
        result.hits().setSource("embedding");
        Hit tensorHit = new Hit("tensor");
        tensorHit.setSource("embedding");
        tensorHit.setField("tensor",embedding);
        result.hits().add(tensorHit);
        return result;
    }

    private Tensor getEmbedding(Query query, Execution execution ){
        Result r = execution.search(query);
        execution.fill(r);
        ErrorHit errorHit = r.hits().getErrorHit();
        if(errorHit != null) {
            throw new RuntimeException(errorHit.toString());
        }
        FeatureData featureData = (FeatureData)r.hits().get(0).getField("summaryfeatures");
        return featureData.getTensor("rankingExpression(mean_token_embedding)").rename("d1","d0");
    }

    private Query getQueryEmbeddingQuery(Query originalQuery)  {
        String queryString = originalQuery.getModel().getQueryString();
        int CLS_TOKEN_ID = 101; //[CLS]
        int SEP_TOKEN_ID = 102; //[SEP]
        int query_max_length = 32;

        List<Integer> token_ids = this.tokenizer.tokenize(queryString,query_max_length,false);
        List<Integer> input_ids = new ArrayList<>(query_max_length);
        List<Integer> attention_mask = new ArrayList<>(query_max_length);

        input_ids.add(CLS_TOKEN_ID);
        input_ids.addAll(token_ids);
        input_ids.add(SEP_TOKEN_ID);
        int length = input_ids.size();
        //Pad up to max length with mask token_id
        int padding = query_max_length - length;

        for(int i = 0; i < length;i++)
            attention_mask.add(1);
        for(int i = 0; i < padding;i++)
            attention_mask.add(0);
        Query query = new Query();
        originalQuery.attachContext(query);
        query.setTimeout("5s");
        query.setHits(1);
        query.getRanking().setProfile("query_embedding");
        query.getModel().setRestrict("query");
        query.getModel().getQueryTree().setRoot(new WordItem("query","sddocname"));
        query.getRanking().getFeatures().put(inputIdsTensorName,toTensor(input_ids));
        query.getRanking().getFeatures().put(attentionMaskTensorName,toTensor(attention_mask));
        return query;
    }

    /**
     * Convert to tensor representation with batch dim
     * @param input the List of token_ids
     * @return tensor representation with batch dim
     */

    private Tensor toTensor(List<Integer> input) {
        Tensor.Builder builder = Tensor.Builder.of(inputType);
        int i = 0;
        for(Integer in:input)  {
            if (i == 32)
                break;
            builder.cell(TensorAddress.of(0,i),in); // 0 is batch dim
            i++;
        }
        return builder.build();
    }
}

