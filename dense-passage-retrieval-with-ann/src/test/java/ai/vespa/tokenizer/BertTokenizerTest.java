// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.tokenizer;

import com.yahoo.language.simple.SimpleLinguistics;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BertTokenizerTest {

    static BertModelConfig bertModelConfig;
    static BertTokenizer tokenizer;

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(128);
        bertModelConfig = builder.build();
        try {
            tokenizer = new BertTokenizer(bertModelConfig, new SimpleLinguistics());
        }catch (Exception e) {
        }
    }

    @Test
    public void test_basic() {
        List<Integer> tokens = tokenizer.tokenize("Why is it that google released BERT and what is the difference between caseed and uncased... BERT?",128);
        Integer[] expectedTokens = {2339, 2003, 2009, 2008, 8224, 2207, 14324, 1998, 2054, 2003, 1996, 4489, 2090, 2553, 2098, 1998, 4895, 28969, 1012, 1012, 1012, 14324, 1029};
        assertArrayEquals(expectedTokens, tokens.toArray());
    }

    @Test
    public void test_token_id_to_string() {
        List<Integer> token_ids = tokenizer.tokenize("Where is Donald Trump Juniors",128);
        List<String> strings = tokenizer.convert2Tokens(token_ids);
        String[] expectedStrings = {"where", "is", "donald", "trump", "juniors"};
        assertArrayEquals(expectedStrings, strings.toArray());
    }

    @Test
    public void test_empty() {
        List<Integer> token_ids = tokenizer.tokenize("", 128);
        assertEquals(0, token_ids.size());
        token_ids = tokenizer.tokenize("", 128, true);
        assertEquals(128, token_ids.size());
    }

    @Test
    public void test_skipping()  {
        StringBuilder buffer = new StringBuilder();
        for(int i = 0; i < 1000; i++)
            buffer.append("foo ");
        String input = buffer.toString();
        List<Integer> token_ids = tokenizer.tokenize(input, 128,true);
        assertEquals(bertModelConfig.max_input(), token_ids.size());
        token_ids = tokenizer.tokenize(input, 128, false);
        assertEquals(128, token_ids.size());

    }

    @Test
    public void test_unicode_simple() {
        String input = "Aaron Aaron \"Ah\u00e4r\u00f4n\" is a prophet,";
        List<Integer> token_ids = tokenizer.tokenize(input,128, false);
        Integer[] expectedTokens = {7158, 7158, 1000, 6289, 10464, 2078, 1000, 2003, 1037, 12168, 1010};
        assertArrayEquals(expectedTokens, token_ids.toArray());
    }

    @Test
    public void test_larger() {
        String passage = "the Indian subcontinent. In the southern hemisphere, the winds are generally milder, but summer storms near Mauritius can be severe." +
                " When the monsoon winds change, cyclones sometimes strike the shores of the Arabian Sea and the Bay of Bengal. The Indian Ocean is the warmest ocean in the world. " +
                "Long-term ocean temperature records show a rapid, continuous warming in the Indian Ocean," +
                " at about during 1901\u20132012. Indian Ocean warming is the largest among the tropical oceans, and about 3 times faster than the warming observed in the Pacific. " +
                "Research indicates that human induced greenhouse warming, and changes in the frequency";

        String question = "why is the indian ocean the warmest in the world";
        List<Integer> passage_token_ids = tokenizer.tokenize(passage,128,false);
        List<Integer> question_token_ids = tokenizer.tokenize(question,128, false);
        Integer[] expected_question_token_ids = {2339,  2003,  1996,  2796,  4153,  1996,  4010,  4355,  1999,
                1996,  2088};
        assertArrayEquals(expected_question_token_ids,question_token_ids.toArray());

        Integer[] expected_passage_token_ids = {1996,  2796, 26125,  1012,
                1999,  1996,  2670, 14130,  1010,  1996,  7266,  2024,  3227, 10256,
                2121,  1010,  2021,  2621, 12642,  2379, 18004,  2064,  2022,  5729,
                1012,  2043,  1996, 19183,  7266,  2689,  1010, 26069,  2823,  4894,
                1996, 13312,  1997,  1996, 13771,  2712,  1998,  1996,  3016,  1997,
                8191,  1012,  1996,  2796,  4153,  2003,  1996,  4010,  4355,  4153,
                1999,  1996,  2088,  1012,  2146,  1011,  2744,  4153,  4860,  2636,
                2265,  1037,  5915,  1010,  7142, 12959,  1999,  1996,  2796,  4153,
                1010,  2012,  2055,  2076,  5775,  1516,  2262,  1012,  2796,  4153,
                12959,  2003,  1996,  2922,  2426,  1996,  5133, 17401,  1010,  1998,
                2055,  1017,  2335,  5514,  2084,  1996, 12959,  5159,  1999,  1996,
                3534,  1012,  2470,  7127,  2008,  2529, 10572, 16635, 12959,  1010,
                1998,  3431,  1999,  1996,  6075};
        assertArrayEquals(expected_passage_token_ids,passage_token_ids.toArray());

    }

    @Test
    public void test_larger_another() {
        Integer[] expected_passage_token_ids = {
                4487, 13102, 19217,  2015,
                1996,  7722,  1006,  1996,  3872,  1997,  2300, 20064,  2003,  1015,
                1010,  6352,  2335,  3618,  2084,  6381,  2300,  1010,  2012,  2009,
                24545,  2058,  1018,  1010,  2199,  2335,  1007,  1012,  2023,  3727,
                1996,  2543,  2302,  2438,  1997,  1996, 22863, 19966,  3512,  4005,
                1010,  1998,  2009,  8289,  2041,  1012,  1996, 20064,  3989,  1997,
                2300,  2036, 16888,  2015,  3684,  1025,  2009,  8558,  4658,  2015,
                1996,  5610,  1010,  2250,  1010,  3681,  1010,  1998,  5200,  2008,
                2071,  2552,  2004,  2582,  4762,  1010,  1998,  2947, 16263,  2028,
                1997,  1996,  2965,  2011,  2029,  8769,  4982,  1010,  2029,  2003,
                2011,  1000,  8660,  1000,  2000,  3518,  3684,  1013,  4762,  4216,
                2000,  2707,  2047,  8769,  1010,  2029,  2059, 11506,  1012,  2300,
                4654,  3436, 27020, 14227,  4765,  2003,  2947,  1037,  5257,  1997,
                1000,  2004, 21281, 14787,  1000,  1006,  6276,  2125,  1996,  7722,
                4425,  1007,  1998, 11520,  1012,  1996,  8457
        };
        String passage = "displaces the oxygen (the volume of water vapor is 1,700 times greater than liquid water, at it expands over 4,000 times). " +
                "This leaves the fire without enough of the combustive agent, and it dies out. The vaporization of water also absorbs heat; it thereby cools " +
                "the smoke, air, walls, and objects that could act as further fuel, and thus prevents one of the means by which fires grow, which is by \"jumping\" to nearby heat/fuel sources to start new fires, " +
                "which then combine. Water extinguishment is thus a combination of \"asphyxia\" (cutting off the oxygen supply) and cooling. The flame";
        List<Integer> passage_token_ids = tokenizer.tokenize(passage,256,false);
        assertArrayEquals(expected_passage_token_ids,passage_token_ids.toArray());

    }

}
