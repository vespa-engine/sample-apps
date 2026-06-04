// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.linguistics.asia.lucene;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Combines consecutive tokens back into a single token when their
 * concatenation appears in a phrase dictionary.
 *
 * <p>Designed to repair over-segmentation caused by {@code icuTokenizer}
 * on CJK input: ICU may split a known compound like {@code 笔记本电脑}
 * into {@code [笔记, 本, 电脑]}, defeating phrase-level lookups. With a
 * dictionary entry {@code 笔记本电脑}, this filter aggregates the three
 * tokens back into one — making downstream synonym filters,
 * keyword-marker filters, etc. see the phrase as a single unit.
 *
 * <p>Algorithm: read all input tokens into a buffer (acceptable for the
 * relatively short text fields typical of search indexes), then scan
 * left-to-right doing longest-match against the dictionary. Match
 * lengths capped by the longest entry observed at load time.
 */
public final class PhraseDictTokenFilter extends TokenFilter {

    private final Set<String> phrases;
    private final int maxPhraseChars;

    private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
    private final OffsetAttribute offsetAtt = addAttribute(OffsetAttribute.class);
    private final PositionIncrementAttribute posAtt = addAttribute(PositionIncrementAttribute.class);

    private boolean initialized = false;
    private List<Emitted> out = new ArrayList<>();
    private int idx = 0;

    private static final class Buffered {
        final String text;
        final int startOffset;
        final int endOffset;
        final int posIncrement;
        Buffered(String text, int startOffset, int endOffset, int posIncrement) {
            this.text = text;
            this.startOffset = startOffset;
            this.endOffset = endOffset;
            this.posIncrement = posIncrement;
        }
    }

    private static final class Emitted {
        final String text;
        final int startOffset;
        final int endOffset;
        final int posIncrement;
        Emitted(String text, int startOffset, int endOffset, int posIncrement) {
            this.text = text;
            this.startOffset = startOffset;
            this.endOffset = endOffset;
            this.posIncrement = posIncrement;
        }
    }

    public PhraseDictTokenFilter(TokenStream input, Set<String> phrases) {
        super(input);
        this.phrases = phrases;
        int max = 0;
        for (String p : phrases) max = Math.max(max, p.length());
        this.maxPhraseChars = Math.max(max, 1);
    }

    @Override
    public boolean incrementToken() throws IOException {
        if (!initialized) {
            initialize();
            initialized = true;
        }
        if (idx >= out.size()) return false;
        Emitted e = out.get(idx++);
        clearAttributes();
        termAtt.setEmpty().append(e.text);
        offsetAtt.setOffset(e.startOffset, e.endOffset);
        posAtt.setPositionIncrement(e.posIncrement);
        return true;
    }

    private void initialize() throws IOException {
        // Drain the entire input stream once.
        List<Buffered> all = new ArrayList<>();
        while (input.incrementToken()) {
            all.add(new Buffered(
                    termAtt.toString(),
                    offsetAtt.startOffset(),
                    offsetAtt.endOffset(),
                    posAtt.getPositionIncrement()));
        }
        // Greedy longest-match phrase scan.
        int i = 0;
        while (i < all.size()) {
            int matchedLen = 0;
            String matched = null;
            int maxSpan = Math.min(all.size() - i, maxPhraseChars);
            for (int n = maxSpan; n >= 2; n--) {
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < n; k++) sb.append(all.get(i + k).text);
                String candidate = sb.toString();
                if (candidate.length() > maxPhraseChars) continue;
                if (phrases.contains(candidate)) {
                    matchedLen = n;
                    matched = candidate;
                    break;
                }
            }
            if (matchedLen >= 2) {
                Buffered first = all.get(i);
                Buffered last = all.get(i + matchedLen - 1);
                out.add(new Emitted(matched, first.startOffset, last.endOffset, first.posIncrement));
                i += matchedLen;
            } else {
                Buffered b = all.get(i);
                out.add(new Emitted(b.text, b.startOffset, b.endOffset, b.posIncrement));
                i += 1;
            }
        }
    }

    @Override
    public void reset() throws IOException {
        super.reset();
        out = new ArrayList<>();
        idx = 0;
        initialized = false;
    }

    /** Used by the factory to build the immutable phrase set. */
    static Set<String> phrasesFrom(List<String> lines) {
        Set<String> s = new HashSet<>();
        for (String raw : lines) {
            if (raw == null) continue;
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            s.add(line);
        }
        return s;
    }

}
