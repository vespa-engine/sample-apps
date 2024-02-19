/*
 * Copyright 2020 Qihoo Corporation.
 * Licensed under the terms of the Apache 2.0 license.
 * See LICENSE in the project root.
 * Author: Tanzhenghai
 */

package com.qihoo.language;

import com.google.inject.Inject;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.qihoo.language.config.JiebaConfig;
import com.yahoo.language.Linguistics;
import com.yahoo.language.opennlp.OpenNlpLinguistics;
import com.yahoo.language.process.Segmenter;
import com.yahoo.language.process.SegmenterImpl;
import com.yahoo.language.process.Stemmer;
import com.yahoo.language.process.StemmerImpl;
import com.yahoo.language.process.Tokenizer;

/**
 * Factory of jieba linguistic processor implementations.
 *
 * @author tanzhenghai 
 */
public class JiebaLinguistics extends OpenNlpLinguistics {


    private final Tokenizer tokenizer;
    private final Tokenizer queryTokenizer;

    @Inject
    public JiebaLinguistics(JiebaConfig config) {
        this.tokenizer = new JiebaTokenizer(config, JiebaSegmenter.SegMode.INDEX);
        this.queryTokenizer = new JiebaTokenizer(config, JiebaSegmenter.SegMode.SEARCH);
    }

    @Override
    public Tokenizer getTokenizer() {
        return tokenizer;
    }

    @Override
    public Stemmer getStemmer() {
        return new StemmerImpl(queryTokenizer);
    }

    @Override
    public Segmenter getSegmenter() {
        return new SegmenterImpl(queryTokenizer);
    }

    @Override
    public boolean equals(Linguistics other) {
        return other instanceof JiebaLinguistics;
    }

}

