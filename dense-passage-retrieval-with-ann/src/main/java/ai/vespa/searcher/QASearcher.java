package ai.vespa.searcher;

import ai.vespa.QuestionAnswering;
import ai.vespa.tokenizer.BertTokenizer;
import ai.vespa.QuestionAnswering.Span;
import com.google.inject.Inject;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.processing.request.CompoundName;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.search.result.FeatureData;




@Provides("Answering")
public class QASearcher extends Searcher {

    BertTokenizer tokenizer;
    private static CompoundName RETRIEVER_ONLY =  new CompoundName("retrieve-only");
    private static int MAX_SEQUENCE_LENGTH = 380;

    @Inject
    public QASearcher(BertTokenizer tokenizer)  {
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        if(query.getModel().getRestrict().contains("query") ||
                query.properties().getBoolean(RETRIEVER_ONLY,false))
            return execution.search(query); //Do nothing - pass it through

        query.setHits(1); //We only extract text spans from the hit with the highest Reader relevancy score
        Result result = execution.search(query);
        execution.fill(result);
        Hit answerHit = getPredictedAnswer(result);
        result.hits().remove(0);
        result.hits().add(answerHit);
        return result;
    }

    /**
     * Get the Result and pick the best span from the best hit
     * @param result the original result
     * @return a formatted Hit which can be rendered
     */
    private Hit getPredictedAnswer(Result result) {
        if(result.getTotalHitCount() == 0 || result.hits().getErrorHit() != null) {
            return formatAnswerHit(null);
        }

        Hit bestReaderHit = result.hits().get(0);
        double readerScore = bestReaderHit.getRelevance().getScore(); //Reader relevancy logit score
        //Final sequence input including special tokens
        Tensor input = getTensor(bestReaderHit, "rankingExpression(input_ids)");
        //The start position logits, relative to the input including special tokens (e.g 101,102)
        Tensor startLogits = getTensor(bestReaderHit, "onnxModel(files_reader_onnx).output_0");
        //The end position logits, relative to the input including special tokens (e.g 101,102)
        Tensor endLogits = getTensor(bestReaderHit, "onnxModel(files_reader_onnx).output_1");
        Span bestSpan = QuestionAnswering.getSpan(startLogits,endLogits,input,MAX_SEQUENCE_LENGTH, readerScore, tokenizer);
        bestSpan.setContext((String)bestReaderHit.getField("text"));
        bestSpan.setContextTitle((String)bestReaderHit.getField("title"));
        return formatAnswerHit(bestSpan);
    }

    private Tensor getTensor(Hit hit, String output) {
        FeatureData featureData = (FeatureData)hit.getField("summaryfeatures");
        return featureData.getTensor(output);
    }

    private Hit formatAnswerHit(Span span) {
        Hit answer = new Hit("answer");
        if(span == null) {
            return answer;
        }
        answer.setField("prediction", span.getPrediction());
        answer.setField("context", span.getContext());
        answer.setField("context_title", span.getTitle());
        answer.setField("prediction_score", span.getSpanScore());
        answer.setField("reader_score", span);
        return answer;
    }


}
