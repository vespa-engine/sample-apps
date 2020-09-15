package ai.vespa.tokenizer;


import org.junit.Test;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import static org.junit.Assert.assertArrayEquals;

public class BertTokenizerTest {

    static BertModelConfig bertModelConfig;

    static {
        BertModelConfig.Builder builder = new BertModelConfig.Builder();
        builder.vocabulary(new com.yahoo.config.FileReference("src/test/resources/bert-base-uncased-vocab.txt")).max_input(128);
        bertModelConfig = builder.build();
    }

    @Test
    public void testBasicTokenizer() throws IOException {
        BertTokenizer tokenizer = new BertTokenizer(bertModelConfig);
        List<Integer> tokens = tokenizer.tokenize("Why is it that google released BERT and what is the difference between caseed and uncased... BERT?");
        Integer[] expectedTokens = {2339, 2003, 2009, 2008, 8224, 2207, 14324, 1998, 2054, 2003, 1996, 4489, 2090, 2553, 2098, 1998, 4895, 28969, 1012, 1012, 1012, 14324, 1029};
        assertArrayEquals(expectedTokens, tokens.toArray());
    }

    @Test
    public void testBackTrack() throws IOException {
        BertTokenizer tokenizer = new BertTokenizer(bertModelConfig);
        List<Integer> token_ids = tokenizer.tokenize("Why is it that google released BERT and what is the difference between caseed and uncased... BERT?");
        Integer[] expectedTokens = {2339, 2003, 2009, 2008, 8224, 2207, 14324, 1998, 2054, 2003, 1996, 4489, 2090, 2553, 2098, 1998, 4895, 28969, 1012, 1012, 1012, 14324, 1029};
        assertArrayEquals(expectedTokens, token_ids.toArray());
        List<String> tokens = tokenizer.convert2Tokens(token_ids);
        System.out.println(tokens);

    }

    @Test
    public void testBasicTokenizerPadding() throws IOException {

        BertTokenizer tokenizer = new BertTokenizer(bertModelConfig);
        List<Integer> tokens = tokenizer.tokenize("Why is it that google released BERT and what is the difference between caseed and uncased... BERT?", true);
        Integer[] realTokens = {2339, 2003, 2009, 2008, 8224, 2207, 14324, 1998, 2054, 2003, 1996, 4489, 2090, 2553, 2098, 1998, 4895, 28969, 1012, 1012, 1012, 14324, 1029};
        List<Integer> expectedWithPadding = new ArrayList<>(128);
        for (int i = 0; i < tokenizer.getMaxLength(); i++) {
            if (i < realTokens.length)
                expectedWithPadding.add(realTokens[i]);
            else
                expectedWithPadding.add(0);

        }
        assertArrayEquals(expectedWithPadding.toArray(), tokens.toArray());
    }

    @Test
    public void longTextTest() throws IOException{
        String input = "Aaron Aaron ( or ; \"Ah\u00e4r\u00f4n\") is a prophet, " +
                "high priest, and the brother of Moses in the Abrahamic religions." +
                " Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. " +
                "The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with " +
                "their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (\"prophet\")" +
                "to the Pharaoh. Part of the Law (Torah) that Moses received from";
        BertTokenizer tokenizer = new BertTokenizer(bertModelConfig);
        List<Integer> token_ids = tokenizer.tokenize(input,true);
        System.out.println(token_ids);

    }
}
