// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

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

    private final BertTokenizer tokenizer;

    @Inject
    public QASearcher(BertTokenizer tokenizer)  {
        this.tokenizer = tokenizer;
    }

    @Override
    public Result search(Query query, Execution execution) {
        if(query.getModel().getRestrict().contains("query") ||
                QuestionAnswering.isRetrieveOnly(query))
            return execution.search(query);

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
            return formatAnswerHit(null,0,0);
        }

        Hit bestReaderHit = result.hits().get(0);
        long id = (long)bestReaderHit.getField("id");
        double readerScore = bestReaderHit.getRelevance().getScore();

        FeatureData featureData = (FeatureData)bestReaderHit.getField("summaryfeatures");

        Tensor input = featureData.getTensor("rankingExpression(input_ids)");
        Tensor startLogits = featureData.getTensor( "onnxModel(reader).start_logits");
        Tensor endLogits = featureData.getTensor("onnxModel(reader).end_logits");
        double firstPhase = featureData.getDouble("firstPhase");
        Span bestSpan = QuestionAnswering.getSpan(startLogits,endLogits,input, readerScore, tokenizer);
        bestSpan.setContext((String)bestReaderHit.getField("text"));
        bestSpan.setContextTitle((String)bestReaderHit.getField("title"));
        return formatAnswerHit(bestSpan,id,firstPhase);
    }

    private Hit formatAnswerHit(Span span, long id, double firstPhase) {
        Hit answer = new Hit("answer");
        if(span == null) {
            return answer;
        }
        answer.setField("prediction", span.getPrediction());
        answer.setField("context", span.getContext());
        answer.setField("context_title", span.getTitle());
        answer.setField("passage_id", id);
        answer.setField("span_score", span.getSpanScore());
        answer.setField("reader_score", span.getReaderScore());
        answer.setField("retriever_score", firstPhase);
        return answer;
    }
}
