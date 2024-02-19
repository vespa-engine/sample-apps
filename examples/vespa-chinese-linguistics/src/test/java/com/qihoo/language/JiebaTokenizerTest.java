package com.qihoo.language;

import com.huaban.analysis.jieba.JiebaSegmenter;
import com.qihoo.language.config.JiebaConfig;
import com.yahoo.config.FileReference;
import com.yahoo.language.Language;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.Iterator;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author bratseth
 */
public class JiebaTokenizerTest {

    @Test
    public void testJiebaTokenizer() {
        String text = "e-tron是Audi生产的车";
        var tokenizer = new JiebaTokenizer(new JiebaConfig.Builder().build(), JiebaSegmenter.SegMode.INDEX);
        Iterator<Token> tokens = tokenizer.tokenize(text, Language.CHINESE_SIMPLIFIED, StemMode.ALL, true).iterator();
        assertToken("e", tokens);
        assertToken("-", tokens);
        assertToken("tron", tokens);
        assertToken("是", tokens);
        assertToken("audi", tokens);
        assertToken("生产", tokens);
        assertToken("的", tokens);
        assertToken("车", tokens);
    }

    @Test
    public void testJiebaTokenizerWithConfig() {
        String text = "my e-tron是Audi生产的车";
        var tokenizer = new JiebaTokenizer(new JiebaConfig.Builder()
                                                   .dictionary(testFile("src/test/resources/dictionary.dict"))
                                                   .stopwords(testFile("src/test/resources/stopwords"))
                                                   .build(),
                                                               JiebaSegmenter.SegMode.INDEX);
        Iterator<Token> tokens = tokenizer.tokenize(text, Language.CHINESE_SIMPLIFIED, StemMode.ALL, true).iterator();
        assertToken(" ", tokens);
        assertToken("e", tokens);
        assertToken("-", tokens);
        assertToken("tron", tokens);
        assertToken("是", tokens);
        assertToken("audi", tokens);
        assertToken("生产", tokens);
        assertToken("的", tokens);
        assertToken("车", tokens);
    }

    private Optional<FileReference> testFile(String path) {
        return Optional.of(FileReference.mockFileReferenceForUnitTesting(new File(path)));
    }

    private void assertToken(String tokenString, Iterator<Token> tokens) {
        assertEquals(tokenString, tokens.next().getTokenString());
    }

}
