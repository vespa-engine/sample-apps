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
import java.util.Set;

/**
 * This is not multithread safe.
 *
 * @author Tanzhenghai 
 */
public class JiebaTokenizer implements Tokenizer {

    private final Set<String> stopwords;
    private final JiebaSegmenter segmenter;

    public JiebaTokenizer(DictsLocConfig config) {
        this.stopwords = readStopwords(config);
        this.segmenter = new JiebaSegmenter();
        if ( ! config.dictionaryPath().isEmpty()) {
            File dictionaryFile = new File(config.dictionaryPath());
            if ( ! dictionaryFile.exists())
                throw new IllegalArgumentException("Failed initializing the Jieba tokenizer: " +
                                                   "Could not read dictionary file '" + dictionaryFile + "'");
            this.segmenter.initUserDict(new String[]{config.dictionaryPath()});
        }
    }

    private Set<String> readStopwords(DictsLocConfig config) {
        if (config.stopwordsPath().isEmpty()) return Set.of();
        File stopwordsFile = new File(config.stopwordsPath());
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(stopwordsFile))) {
            Set<String> stopwords = new HashSet<>();
            String temp;
            while ((temp = bufferedReader.readLine()) != null)
                stopwords.add(temp.trim());
            return Collections.unmodifiableSet(stopwords);
        }
        catch (IOException e) {
            throw new IllegalArgumentException("Failed initializing the Jieba tokenizer: " +
                                               "Could not read dictionary file '" + stopwordsFile + "'", e);
        }
    }

    @Override
    public Iterable<Token> tokenize(String input, Language language, StemMode stemMode, boolean removeAccents) {
        if (input.isEmpty()) return List.of();

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
