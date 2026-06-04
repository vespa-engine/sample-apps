// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.linguistics.asia;

import com.yahoo.language.Language;
import com.yahoo.language.opennlp.OpenNlpLinguistics;
import com.yahoo.language.process.LinguisticsParameters;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class IcuTokenizerTest {

    private static List<String> tokenStringsOf(Iterable<Token> tokens) {
        List<String> out = new ArrayList<>();
        for (Token t : tokens) out.add(t.getTokenString());
        return out;
    }

    private static List<String> origsOf(Iterable<Token> tokens) {
        List<String> out = new ArrayList<>();
        for (Token t : tokens) out.add(t.getOrig());
        return out;
    }

    @Test
    void traditionalAndSimplifiedTokenStringsMatch() {
        IcuTokenizer tokenizer = new IcuTokenizer(new OpenNlpLinguistics().getTokenizer());
        LinguisticsParameters simp = new LinguisticsParameters(Language.CHINESE_SIMPLIFIED, StemMode.NONE, false, false);
        LinguisticsParameters trad = new LinguisticsParameters(Language.CHINESE_TRADITIONAL, StemMode.NONE, false, false);

        assertEquals(
                tokenStringsOf(tokenizer.tokenize("繁体中文", simp)),
                tokenStringsOf(tokenizer.tokenize("繁體中文", trad)),
                "Traditional input must produce the same token strings as its Simplified equivalent");
    }

    @Test
    void getOrigKeepsOriginalScriptGetTokenStringFolds() {
        // Critical invariant: orig != tokenString for Traditional input.
        // Vespa's LinguisticsAnnotator interprets orig == tokenString as
        // "no override; index the original field substring", which would
        // defeat the fold at index time.
        IcuTokenizer tokenizer = new IcuTokenizer(new OpenNlpLinguistics().getTokenizer());
        LinguisticsParameters trad = new LinguisticsParameters(Language.CHINESE_TRADITIONAL, StemMode.NONE, false, false);

        Iterable<Token> tokens = tokenizer.tokenize("蘋果", trad);
        List<String> origs = origsOf(tokens);
        List<String> strs  = tokenStringsOf(tokens);

        assertEquals(List.of("蘋果"), origs, "getOrig() must preserve the original Traditional substring");
        assertEquals(List.of("苹果"), strs,  "getTokenString() must be the Simplified form");
        assertNotEquals(origs, strs, "orig and tokenString must differ — equality triggers Vespa's null-indexed-form optimization");
    }

}
