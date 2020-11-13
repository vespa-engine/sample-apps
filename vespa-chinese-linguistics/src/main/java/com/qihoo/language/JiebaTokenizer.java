// Copyright 2020 Qihoo Corporation. Licensed under the terms of the Apache 2.0 license. 
// Author: Tanzhenghai
//
package com.qihoo.language;

import com.qihoo.language.config.DictsLocConfig;
import com.yahoo.language.Language;
import com.yahoo.language.process.StemMode;
import com.yahoo.language.process.Token;
import com.yahoo.language.process.Tokenizer;
import com.yahoo.language.simple.SimpleToken;
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
import java.io.IOException;

/**
 * This is not multithread safe.
 *
 * @author Tanzhenghai 
 */
public class JiebaTokenizer implements Tokenizer {

    private final JiebaSegmenter segmenter;
    private final HashSet<String> stopwords = new HashSet<>();

    public JiebaTokenizer(DictsLocConfig cfg)  {
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
                    String temp;
                    while ((temp = bufferedReader.readLine()) != null) {
                        stopwords.add(temp.trim());
                    }
                }
                catch (IOException e) {
                }
            }
        }
    }

    @Override
    public Iterable<Token> tokenize(String input, Language language, StemMode stemMode, boolean removeAccents) {
        if (input.isEmpty()) return Collections.emptyList();

        List<Token> tokens = new ArrayList<>();
        for (SegToken token : segmenter.process(input, SegMode.INDEX)){
            if (stopwords.contains(token.word)) continue;
            tokens.add(new SimpleToken(token.word).setOffset(token.startOffset)
                                                  .setType(TokenType.ALPHABETIC)
                                                  .setTokenString(token.word));
        }	
        return tokens;
    }

}
