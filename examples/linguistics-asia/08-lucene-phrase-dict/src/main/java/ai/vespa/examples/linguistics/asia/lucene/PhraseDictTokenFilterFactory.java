// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.examples.linguistics.asia.lucene;

import org.apache.lucene.analysis.TokenFilterFactory;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.util.ResourceLoader;
import org.apache.lucene.util.ResourceLoaderAware;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Lucene SPI factory for {@link PhraseDictTokenFilter}. Reads a phrase
 * dictionary from {@code configDir} (one phrase per line, '#'-comments
 * allowed) at component-construction time.
 *
 * <p>Plug a real Chinese phrase dictionary (Academia Sinica CKIP,
 * HanLP, IK, Jieba user-dict, or any in-house list) by replacing
 * {@code dictionary.txt} in the application package — the factory
 * accepts any text file in this format.
 */
public class PhraseDictTokenFilterFactory extends TokenFilterFactory implements ResourceLoaderAware {

    public static final String NAME = "phraseDict";

    private final String resource;
    private Set<String> phrases = Collections.emptySet();

    public PhraseDictTokenFilterFactory(Map<String, String> args) {
        super(args);
        this.resource = require(args, "dictionary");
        if (!args.isEmpty()) throw new IllegalArgumentException("Unknown parameters: " + args);
    }

    public PhraseDictTokenFilterFactory() {
        super(new java.util.HashMap<>());
        this.resource = null;
    }

    @Override
    public void inform(ResourceLoader loader) throws IOException {
        if (resource == null) return;
        List<String> lines = new ArrayList<>();
        try (InputStream in = loader.openResource(resource);
             BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) lines.add(line);
        }
        this.phrases = PhraseDictTokenFilter.phrasesFrom(lines);
    }

    @Override
    public TokenStream create(TokenStream input) {
        return new PhraseDictTokenFilter(input, phrases);
    }

}
