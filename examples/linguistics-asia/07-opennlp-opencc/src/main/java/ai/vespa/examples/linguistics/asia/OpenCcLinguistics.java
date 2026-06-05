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
 * Custom Linguistics that folds Traditional Chinese (including
 * Hong Kong and Taiwan orthography variants) to Simplified using
 * OpenCC4j's directional converters. zh-CN, zh-TW, and zh-HK
 * content all normalize to Simplified so a single index serves
 * all three markets.
 *
 * <p>Same wrapping pattern as
 * {@link ai.vespa.examples.linguistics.asia.IcuLinguistics} in 05.
 * The library swap (ICU4J -> OpenCC4j) is the only change.
 */
public class OpenCcLinguistics extends OpenNlpLinguistics {

    private final Tokenizer tokenizer;

    public OpenCcLinguistics() {
        super();
        this.tokenizer = new OpenCcTokenizer(super.getTokenizer());
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
        return other instanceof OpenCcLinguistics;
    }

}
