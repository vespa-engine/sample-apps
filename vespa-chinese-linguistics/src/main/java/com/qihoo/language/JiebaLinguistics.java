/*
 * Copyright 2020 Qihoo Corporation.
 * Licensed under the terms of the Apache 2.0 license.
 * See LICENSE in the project root.
 * Author: Tanzhenghai
 */

package com.qihoo.language.jieba;

import com.google.inject.Inject;
import com.yahoo.collections.Tuple2;
import com.yahoo.component.Version;
import com.yahoo.language.Linguistics;
import com.yahoo.language.detect.Detector;
import com.yahoo.language.process.CharacterClasses;
import com.yahoo.language.process.GramSplitter;
import com.yahoo.language.process.Normalizer;
import com.yahoo.language.process.Segmenter;
import com.yahoo.language.process.SegmenterImpl;
import com.yahoo.language.process.Stemmer;
import com.yahoo.language.process.StemmerImpl;
import com.yahoo.language.process.Tokenizer;
import com.yahoo.language.process.Transformer;
import com.yahoo.language.simple.SimpleNormalizer;
import com.yahoo.language.simple.SimpleTransformer;
import com.yahoo.language.simple.SimpleToken;
import com.yahoo.language.simple.SimpleDetector;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * Factory of jieba linguistic processor implementations.
 * Useful for 
 *
 * @author tanzhenghai 
 */
public class JiebaLinguistics implements Linguistics {

    // Threadsafe instances
    private final Normalizer normalizer;
    private final Transformer transformer;
    private final Detector detector;
    private final CharacterClasses characterClasses;
    private final GramSplitter gramSplitter;
    private final Tokenizer tokenizer;
    private final DictsLocConfig cfg;
private static final Logger log = Logger.getLogger(JiebaLinguistics.class.getName());

    @Inject
    public JiebaLinguistics(DictsLocConfig config) {
        this.normalizer = new SimpleNormalizer();
        this.transformer = new SimpleTransformer();
        this.detector = new SimpleDetector();
        this.characterClasses = new CharacterClasses();
        this.gramSplitter = new GramSplitter(characterClasses);
        this.cfg = config;
        this.tokenizer = new JiebaTokenizer(config, normalizer, transformer);
    }

    @Override
    public Stemmer getStemmer() { return new StemmerImpl(getTokenizer()); }

    @Override
    public Tokenizer getTokenizer() { return tokenizer; }

    @Override
    public Normalizer getNormalizer() { return normalizer; }

    @Override
    public Transformer getTransformer() { return transformer; }

    @Override
    public Segmenter getSegmenter() { return new SegmenterImpl(getTokenizer()); }

    @Override
    public Detector getDetector() { return detector; }

    @Override
    public GramSplitter getGramSplitter() { return gramSplitter; }

    @Override
    public CharacterClasses getCharacterClasses() { return characterClasses; }

}

