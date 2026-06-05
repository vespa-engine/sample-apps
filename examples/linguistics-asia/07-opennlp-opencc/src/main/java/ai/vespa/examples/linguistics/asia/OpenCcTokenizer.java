// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.linguistics.asia;

import com.github.houbb.opencc4j.util.ZhConverterUtil;
import com.yahoo.language.Language;
import com.yahoo.language.process.LinguisticsParameters;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.process.Tokenizer;
import com.yahoo.language.simple.SimpleToken;

import java.util.ArrayList;
import java.util.List;

/**
 * Tokenizer that delegates segmentation to OpenNLP then folds each
 * token's text to Simplified via OpenCC4j. {@code getOrig()} preserves
 * the original Traditional script for the field-text reference; only
 * {@code getTokenString()} carries the folded form, which is what
 * Vespa indexes.
 *
 * <p>To target a specific Chinese variant direction (e.g. Hong Kong
 * Traditional -> Simplified instead of generic Traditional -> Simplified),
 * swap {@code ZhConverterUtil.toSimple} for an explicit OpenCC4j
 * converter instance — see the README for examples.
 */
public class OpenCcTokenizer implements Tokenizer {

    private final Tokenizer delegate;

    public OpenCcTokenizer(Tokenizer delegate) {
        this.delegate = delegate;
    }

    private static String fold(String input) {
        if (input == null || input.isEmpty()) return input;
        return ZhConverterUtil.toSimple(input);
    }

    private Iterable<Token> wrap(Iterable<Token> raw) {
        List<Token> out = new ArrayList<>();
        for (Token t : raw) {
            String origText = t.getOrig();
            String folded   = fold(origText);
            // If fold produced no change (e.g. Latin tokens), preserve the parent
            // tokenizer's tokenString — which already carries lowercasing / etc.
            // applied by OpenNLP. Overriding it here would lose that.
            String tokStr = folded.equals(origText) ? t.getTokenString() : folded;
            SimpleToken st = new SimpleToken(origText)
                    .setOffset(t.getOffset())
                    .setType(t.getType())
                    .setTokenString(tokStr);
            out.add(st);
        }
        return out;
    }

    @Override
    @SuppressWarnings("deprecation")
    public Iterable<Token> tokenize(String input, Language language, StemMode stemMode, boolean removeAccents) {
        return wrap(delegate.tokenize(input, language, stemMode, removeAccents));
    }

    @Override
    public Iterable<Token> tokenize(String input, LinguisticsParameters parameters) {
        return wrap(delegate.tokenize(input, parameters));
    }

}
