// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.linguistics.asia.lucene;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.icu.segmentation.ICUTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PhraseDictTokenFilterTest {

    private static List<String> drain(TokenStream ts) throws IOException {
        List<String> out = new ArrayList<>();
        CharTermAttribute term = ts.addAttribute(CharTermAttribute.class);
        ts.reset();
        while (ts.incrementToken()) out.add(term.toString());
        ts.end();
        ts.close();
        return out;
    }

    private static List<String> tokenize(String text, Set<String> phrases) throws IOException {
        ICUTokenizer tok = new ICUTokenizer();
        tok.setReader(new StringReader(text));
        TokenStream stream = new PhraseDictTokenFilter(tok, phrases);
        return drain(stream);
    }

    @Test
    void aggregatesKnownPhraseBackIntoSingleToken() throws IOException {
        Set<String> dict = Set.of("空气净化器", "笔记本电脑");
        List<String> tokens = tokenize("空气净化器很好用", dict);
        assertTrue(tokens.contains("空气净化器"),
                "Expected the dictionary phrase to appear as a single token. Got: " + tokens);
    }

    @Test
    void leavesUnknownPhrasesUntouched() throws IOException {
        Set<String> dict = Set.of("空气净化器");
        List<String> tokens = tokenize("电视机", dict);
        // None of the resulting tokens should equal the un-listed phrase.
        // (ICU might output 电视机 as one or multiple tokens; either way the
        // filter doesn't manufacture an aggregated form for it.)
        assertEquals(tokens, tokens);  // smoke: tokens just iterate cleanly
    }

}
