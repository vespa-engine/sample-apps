// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.linguistics.asia;

import com.ibm.icu.text.Transliterator;
import com.yahoo.language.Language;
import com.yahoo.language.process.LinguisticsParameters;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.process.Tokenizer;
import com.yahoo.language.simple.SimpleToken;

import java.util.ArrayList;
import java.util.List;

/**
 * Tokenizer that segments via the delegate (OpenNLP) on the ORIGINAL input,
 * then per-token folds Traditional -> Simplified via ICU and stores the
 * folded form in {@code getTokenString()} while leaving {@code getOrig()}
 * as the original substring.
 *
 * <p><b>Why this shape</b>: Vespa's LinguisticsAnnotator stores
 * {@code (offset, orig.length, indexed-form)} per token, and *optimizes*
 * {@code indexed-form} to {@code null} when {@code tokenString.equals(orig)}
 * — meaning "index the original field substring at this offset". If the
 * normalization happens on the whole input before tokenize, both
 * {@code orig} and {@code tokenString} reflect the normalized form, they
 * compare equal, Vespa stores {@code null}, and then re-slices the
 * ORIGINAL field text (which still has Traditional code points). Result:
 * the index gets Traditional terms despite the tokenizer "folding".
 *
 * <p>By keeping {@code orig} = original substring (Traditional) and
 * {@code tokenString} = folded form (Simplified), the two differ and
 * Vespa stores our folded value as the indexed term. Same pattern as
 * {@code JiebaTokenizer}.
 */
public class IcuTokenizer implements Tokenizer {

    private static final Transliterator T2S = Transliterator.getInstance("Traditional-Simplified");

    private final Tokenizer delegate;

    public IcuTokenizer(Tokenizer delegate) {
        this.delegate = delegate;
    }

    private static String fold(String input) {
        if (input == null || input.isEmpty()) return input;
        return T2S.transliterate(input);
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
