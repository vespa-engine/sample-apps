package ai.vespa.searcher;

import ai.vespa.tokenizer.BertTokenizer;
import com.google.inject.Inject;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;

import java.util.List;

public class QASearcher extends Searcher {

    BertTokenizer tokenizer;

    @Inject
    public QASearcher(BertTokenizer tokenizer)  {
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        if(query.getModel().getQueryString() == null ||
                query.getModel().getQueryString().length() == 0)
            return new Result(query, ErrorMessage.createBadRequest("No query input"));
        Tensor questionTensor = getTensorFromQueryText(query.getModel().getQueryString());
        query.getRanking().getFeatures().put("query(query_token_ids)", questionTensor);
        Result result = execution.search(query);
        execution.fill(result);
        return getPredictedAnswer(result);
    }


    private Tensor getTensorFromQueryText(String queryInput) {
        List<Integer> tokens_ids = tokenizer.tokenize(queryInput, true);
        String tensorSpec = "tensor<float>(d0[1],d1[" + tokenizer.getMaxLength() + "]):[" +  tokens_ids + "]";
        return Tensor.from(tensorSpec);
    }

    private Result getPredictedAnswer(Result result) {
        if(result.getTotalHitCount() == 0)
           return result;
        Hit metaHit = new Hit("answer");
        Hit best = result.hits().get(0);
        return result;
    }
}
