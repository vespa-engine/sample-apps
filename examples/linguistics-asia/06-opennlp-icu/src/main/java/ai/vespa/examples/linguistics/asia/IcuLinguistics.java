// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.linguistics.asia;

import com.yahoo.language.Linguistics;
import com.yahoo.language.opennlp.OpenNlpLinguistics;
import com.yahoo.language.process.Segmenter;
import com.yahoo.language.process.SegmenterImpl;
import com.yahoo.language.process.Stemmer;
import com.yahoo.language.process.StemmerImpl;
import com.yahoo.language.process.Tokenizer;

/**
 * Custom Linguistics that folds Traditional Chinese to Simplified at
 * tokenization time so zh-CN and zh-TW content + queries match.
 *
 * <p>Wrapping pattern mirrors vespa-chinese-linguistics (Jieba):
 * build the wrapped tokenizer once and return it from
 * {@link #getTokenizer()}, {@link #getStemmer()}, and
 * {@link #getSegmenter()} so both index-side and query-side Vespa
 * pipelines see the same wrapped tokenizer.
 */
public class IcuLinguistics extends OpenNlpLinguistics {

    private final Tokenizer tokenizer;

    public IcuLinguistics() {
        super();
        this.tokenizer = new IcuTokenizer(super.getTokenizer());
    }

    @Override
    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    @Override
    public Stemmer getStemmer() {
        return new StemmerImpl(tokenizer);
    }

    @Override
    public Segmenter getSegmenter() {
        return new SegmenterImpl(tokenizer);
    }

    @Override
    public boolean equals(Linguistics other) {
        return other instanceof IcuLinguistics;
    }

}
