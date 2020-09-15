package ai.vespa.tokenizer;


import com.yahoo.language.simple.SimpleLinguistics;
import org.junit.Test;
import java.util.List;

import static org.junit.Assert.*;

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
        List<Integer> tokens = tokenizer.tokenize("Why is it that google released BERT and what is the difference between caseed and uncased... BERT?");
        Integer[] expectedTokens = {2339, 2003, 2009, 2008, 8224, 2207, 14324, 1998, 2054, 2003, 1996, 4489, 2090, 2553, 2098, 1998, 4895, 28969, 1012, 1012, 1012, 14324, 1029};
        assertArrayEquals(expectedTokens, tokens.toArray());
    }

    @Test
    public void test_token_id_to_string() {
        List<Integer> token_ids = tokenizer.tokenize("Where is Donald Trump Juniors");
        List<String> strings = tokenizer.convert2Tokens(token_ids);
        String[] expectedStrings = {"where", "is", "donald", "trump", "juniors"};
        assertArrayEquals(expectedStrings, strings.toArray());
    }

    @Test
    public void test_empty() {
        List<Integer> token_ids = tokenizer.tokenize("");
        assertEquals(0, token_ids.size());
        token_ids = tokenizer.tokenize("", true);
        assertEquals(bertModelConfig.max_input(), token_ids.size());
    }

    @Test
    public void test_skipping()  {
        StringBuffer buffer = new StringBuffer();
        for(int i = 0; i < 1000; i++)
            buffer.append("foo ");
        String input = buffer.toString();
        List<Integer> token_ids = tokenizer.tokenize(input, true);
        assertEquals(bertModelConfig.max_input(), token_ids.size());
        token_ids = tokenizer.tokenize(input, false);
        assertEquals(bertModelConfig.max_input(), token_ids.size());

    }

    @Test
    public void test_utf8() {
        String input = "Aaron Aaron \"Ah\u00e4r\u00f4n\" is a prophet,";
        List<Integer> token_ids = tokenizer.tokenize(input,false);
        Integer[] expectedTokens = {7158, 7158, 1000, 6289, 10464, 2078, 1000, 2003, 1037, 12168, 1010};
        assertArrayEquals(expectedTokens, token_ids.toArray());
    }

}
