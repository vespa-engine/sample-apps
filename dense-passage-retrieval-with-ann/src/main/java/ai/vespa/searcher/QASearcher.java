// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.searcher;

import ai.vespa.QuestionAnswering;
import ai.vespa.tokenizer.BertTokenizer;
import ai.vespa.QuestionAnswering.Span;
import com.google.inject.Inject;
import com.yahoo.component.chain.dependencies.Provides;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.tensor.Tensor;
import com.yahoo.search.result.FeatureData;


@Provides("QuestionAnswering")
public class QASearcher extends Searcher {

    BertTokenizer tokenizer;

    @Inject
    public QASearcher(BertTokenizer tokenizer)  {
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        if(query.getModel().getRestrict().contains("query") ||
                QuestionAnswering.isRetrieveOnly(query))
            return execution.search(query);

        query.getRanking().getProperties().put("vespa.hitcollector.heapsize",query.getHits());
        query.setHits(1); //We only extract text spans from the hit with the highest Reader relevancy score

        Result result = execution.search(query);
        execution.fill(result);
        Hit answerHit = getPredictedAnswer(result);
        result.hits().remove(0); //remove original back end hit
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
        double readerScore = bestReaderHit.getRelevance().getScore();

        Tensor input = getTensor(bestReaderHit, "input_ids");
        Tensor startLogits = getTensor(bestReaderHit, "onnx(reader).start_logits");
        Tensor endLogits = getTensor(bestReaderHit, "onnx(reader).end_logits");
        Span bestSpan = QuestionAnswering.getSpan(startLogits,endLogits,input, readerScore, tokenizer);
        bestSpan.setContext((String)bestReaderHit.getField("text"));
        bestSpan.setContextTitle((String)bestReaderHit.getField("title"));
        return formatAnswerHit(bestSpan);
    }

    private Tensor getTensor(Hit hit, String output) {
        FeatureData featureData = (FeatureData)hit.getField("matchfeatures");
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
        answer.setField("reader_score", span.getReaderScore());
        return answer;
    }
}
