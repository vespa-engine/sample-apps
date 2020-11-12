// Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// Author: Tanzhenghai
//
package com.qihoo.language.jieba;

import com.yahoo.language.Language;
import com.yahoo.language.LinguisticsCase;
import com.yahoo.language.process.*;
import com.yahoo.language.simple.SimpleNormalizer;
import com.yahoo.language.simple.SimpleTransformer;
import com.yahoo.language.simple.SimpleToken;
import com.yahoo.language.simple.kstem.KStemmer;
import com.yahoo.language.process.TokenType;
import com.huaban.analysis.jieba.SegToken;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.JiebaSegmenter.SegMode;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.HashSet;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.io.IOException;

/**
 *
 * <p>This is not multithread safe.</p>
 *
 * @author Tanzhenghai 
 */
public class JiebaTokenizer implements Tokenizer {

    private final static int SPACE_CODE = 32;
    private final Normalizer normalizer;
    private final Transformer transformer;
    //private final KStemmer stemmer = new KStemmer();
    private final DictsLocConfig cfg;
    private final JiebaSegmenter segmenter;
    private final HashSet<String> stopwords = new HashSet<String>();
    private static final Logger log = Logger.getLogger(JiebaTokenizer.class.getName());

    public JiebaTokenizer(DictsLocConfig dict) {
        this(dict, new SimpleNormalizer(), new SimpleTransformer());
    }

    public JiebaTokenizer(DictsLocConfig dict, Normalizer normalizer) {
        this(dict, normalizer, new SimpleTransformer());
    }

    public JiebaTokenizer(DictsLocConfig cfg, Normalizer normalizer, Transformer transformer)  {
        this.cfg = cfg;
        this.normalizer = normalizer;
        this.transformer = transformer;
        this.segmenter = new JiebaSegmenter();
        String dict = cfg.dict();
        if (dict.length() > 0){
            File file = new File(dict);
            if (file.exists()) {
                String[] p = new String[]{dict};
                this.segmenter.initUserDict(p);
            }
        }
        dict = cfg.stopwords();
        if (dict.length() > 0){
            File file = new File(dict);
            if (file.exists()) {
                try{
                    BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
                    String temp = null;
                    while ((temp = bufferedReader.readLine()) != null) {
                        stopwords.add(temp.trim());
                    }
                }
                catch (IOException e) {}
            }
        }
    }

    @Override
    public Iterable<Token> tokenize(String input, Language language, StemMode stemMode, boolean removeAccents) {
        if (input.isEmpty()) return Collections.emptyList();

        List<Token> tokens = new ArrayList<>();

        List<SegToken> tks = segmenter.process(input, SegMode.INDEX);
        for (SegToken tk : tks){
            if (stopwords.contains(tk.word)) continue;
            tokens.add(new SimpleToken(tk.word).setOffset(tk.startOffset)
                                            .setType(TokenType.ALPHABETIC)
                                            .setTokenString(tk.word));
        }	

        log.log(Level.FINE, () -> "final tokens are: "+ tokens);
        return tokens;
    }

    private String processToken(String token, Language language, StemMode stemMode, boolean removeAccents) {
        return "";
    }

}
