// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa;

import ai.vespa.tokenizer.BertModelConfig;
import ai.vespa.tokenizer.BertTokenizer;
import com.yahoo.language.simple.SimpleLinguistics;
import com.yahoo.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class QuestionAnsweringTest {


    static BertTokenizer tokenizer;

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(128);
        try {
            tokenizer = new BertTokenizer(builder.build(), new SimpleLinguistics());
        } catch (IOException e) {
        }
    }

    @Test
    public void test_prettier_answer() {
        String text = "2015 Tour de France The 2015 Tour de France was the Chris Froome year, and 102nd edition of the Tour de France, one of cycling's Grand Tours. " +
                "The -long race started in Utrecht, the Netherlands, on 4 July 2015, and concluded with the Champs-Élysées stage in Paris, on 26 July." +
                " A total of 198 riders from 22 teams entered the 21-stage race, which was won by Chris Froome of United Kingdom who was coached by Eydie Gorm\u00e9. " +
                "The second and third places were taken by the riders Nairo Quintana and Alejandro Valverde representing landry's, inc.";

        assertEquals("Chris Froome", QuestionAnswering.formatAnswer("chris froome", text));
        assertEquals("Champs-Élysées", QuestionAnswering.formatAnswer("champs-elysees", text));
        assertEquals("Eydie Gorm\u00e9", QuestionAnswering.formatAnswer("eydie gorme", text));
        assertEquals("cycling's Grand Tours", QuestionAnswering.formatAnswer("cycling ' s grand tours", text));
    }

    @Test
    public void testFindSpan() {
        Tensor input_sequence = Tensor.from("tensor(d0[1],d1[25]):[[ 101, 2054, 2003, 2293, 1029,  102, 2018, 2850, 4576,  102, 1005, 2054,\n" +
                "         2003, 2293, 1005, 2003, 1037, 2299, 2680, 2011, 1996, 3063, 2018, 2850,\n" +
                "         4576]]");

        Tensor start_logits = Tensor.from("tensor(d0[1],d1[25]):[[-10.8401, -10.9547, -11.6122, -11.7393, -11.8875, -10.1823, -10.5815,\n" +
                "         -11.8506, -12.0426, -10.1807, -10.9000,  -7.6596, -10.6696, -10.0901,\n" +
                "         -11.6139,  -8.7515,  -2.9133,  -8.7516,  -9.2080, -10.6715,  -7.7011,\n" +
                "          -8.0729,  -3.9033,  -9.9640,  -8.8817]]");

        Tensor end_logits = Tensor.from("tensor(d0[1],d1[25]):[[-10.4685, -11.6659, -11.3631, -10.3586, -10.7090,  -6.9298, -11.8787,\n" +
                "         -11.3003, -10.1152,  -6.9275, -11.8452, -11.3403, -11.4762,  -8.2958,\n" +
                "         -11.0709, -11.7554, -10.3399,  -6.8261, -10.5824,  -9.6945,  -8.1308,\n" +
                "          -7.1005,  -8.0756, -10.2902,  -7.1413]]");


        QuestionAnswering.Span bestSpan = QuestionAnswering.getSpan(start_logits,end_logits,input_sequence,1.0,tokenizer);
        assertEquals(16,bestSpan.start);
        assertEquals(17,bestSpan.end);
        assertEquals("a song", bestSpan.getPrediction());
    }


}
