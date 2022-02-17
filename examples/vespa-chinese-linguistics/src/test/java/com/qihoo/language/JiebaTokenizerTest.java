package com.qihoo.language;

import com.qihoo.language.config.DictsLocConfig;
import com.yahoo.language.Language;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.assertEquals;

/**
 * @author bratseth
 */
public class JiebaTokenizerTest {

    @Test
    public void testJiebaTokenizer() {
        String text = "e-tron是Audi生产的车";
        var tokenizer = new JiebaTokenizer(new DictsLocConfig.Builder().build());
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

    private void assertToken(String tokenString, Iterator<Token> tokens) {
        assertEquals(tokenString, tokens.next().getTokenString());
    }

}
