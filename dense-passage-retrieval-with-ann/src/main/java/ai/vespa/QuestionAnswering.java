// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa;


import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.search.Query;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class QuestionAnswering {

    private static int MAX_ANSWER_LENGTH = 10;

    public enum RetrievalMethod {
        SPARSE,
        DENSE,
        HYBRID;
    }

    public static class Span implements Comparable<Span>{
        int start;
        int end;
        Double spanScore;
        Double readerScore;
        String prediction;


        String context;
        String title;

        Span(int start, int end, double spanScore, double readerScore)  {
            this.start = start;
            this.end = end;
            this.spanScore = spanScore;
            this.readerScore = readerScore;
        }

        @Override
        public int compareTo(Span other) {
            return other.spanScore.compareTo(this.spanScore);
        }

        @Override
        public String toString()  {
            return "start=" + start + ",end=" + end + ",span_score=" + spanScore + ", answer=" + getPrediction();
        }

        public double getSpanScore() {
            return this.spanScore;
        }

        public double getReaderScore() {
            return readerScore;
        }

        public void setPrediction(String prediction) {
            this.prediction = prediction;
        }
        public String getPrediction() {
            return formatAnswer(this.prediction, context);
        }
        public void setContext(String context) {
            this.context = context;
        }
        public void setContextTitle(String title) {
            this.title = title;
        }
        public String getContext() {
            return context;
        }

        public String getTitle() {
            return title;
        }
    }

    static public Span getSpan(Tensor startLogits, Tensor endLogits, Tensor input, double readerScore,
                               BertTokenizer tokenizer)  {
        long inputLength = startLogits.size();
        List<Span> spans = new ArrayList<>();
        for(int i = 0; i < inputLength; i++) {
            double startScore = startLogits.get(TensorAddress.of(0,i));
            long maxLength = Math.min(i+MAX_ANSWER_LENGTH,inputLength);
            for(int j = i; j < maxLength; j++)  {
                double endScore = endLogits.get(TensorAddress.of(0,j));
                spans.add(new Span(i,j,startScore + endScore, readerScore));
            }
        }
        Collections.sort(spans);
        Span bestSpan = spans.get(0);
        ArrayList<Integer> token_id_span = new ArrayList<>();
        int start = bestSpan.start;
        int end = bestSpan.end;

        while(start > 0 && tokenizer.isSubWord(tokenizer.getTokenFromId((int)input.get(TensorAddress.of(0,start)))))
            start--;
        while(end < inputLength -1 && tokenizer.isSubWord(tokenizer.getTokenFromId((int)input.get(TensorAddress.of(0,end +1)))))
            end++;

        for(int i = start; i <= end ;i++) {
            Integer token_id = (int)input.get(TensorAddress.of(0,i));
            token_id_span.add(token_id);
        }
        String answer = tokenizer.removeSubWords(tokenizer.convert2Tokens(token_id_span));
        bestSpan.setPrediction(answer);
        return bestSpan;
    }

    /**
     *Tries to find the original casing from the context, also attemps
     * to restore accent which are removed from the vocab e => é
     * @param prediction The vocab answer
     * @param context The original text snippet the answer is from
     * @return the original string is found, if not returns the original prediction text
     */

    static String formatAnswer(String prediction, String context) {
        prediction = prediction.replace(" ' ","'");
        prediction = prediction.replace(" , ",", ");
        prediction = prediction.replace(" . ",". ");

        if(context == null)
            return prediction;

        Pattern pattern = Pattern.compile(Pattern.quote(prediction),
                Pattern.CASE_INSENSITIVE | Pattern.MULTILINE);
        String normalized_context = java.text.Normalizer.normalize(context,
                Normalizer.Form.NFKD).replaceAll("\\p{InCombiningDiacriticalMarks}+", "");
        Matcher m = pattern.matcher(normalized_context);
        String formattedAnswer = prediction;
        if (m.find()) {
            int start = m.start();
            int end = m.end();
            int length = context.length();
            if(start < length && end <= length)
                return context.substring(start, end);
        }
        return formattedAnswer;
    }

    public static boolean isRetrieveOnly(Query query) {
        return query.properties().getBoolean("retrieve-only",false);
    }

    public static RetrievalMethod getRetrivalMethod(Query query)  {
       String model = query.properties().getString("retriever","dense");
       if(model.equals("dense"))
           return RetrievalMethod.DENSE;
       else if(model.equals("sparse"))
           return RetrievalMethod.SPARSE;
       else return RetrievalMethod.HYBRID;
    }
}
