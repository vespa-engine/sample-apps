package ai.vespa.linguistics.lemmagen;

import eu.hlavki.text.lemmagen.LemmatizerFactory;
import eu.hlavki.text.lemmagen.api.Lemmatizer;
import org.apache.lucene.analysis.TokenFilterFactory;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.util.ResourceLoader;
import org.apache.lucene.util.ResourceLoaderAware;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

/**
 * https://lucene.apache.org/core/9_7_0/
 * https://github.com/vhyza/elasticsearch-analysis-lemmagen
 * Loosely based on
 * https://github.com/vhyza/elasticsearch-analysis-lemmagen/blob/master/src/main/java/org/elasticsearch/index/analysis/LemmagenFilterFactory.java
 * Also inspired by
 * https://github.com/hlavki/jlemmagen-lucene/blob/master/src/main/java/org/apache/lucene/analysis/lemmagen/LemmagenFilterFactory.java
 */
public class LemmagenTokenFilterFactory extends TokenFilterFactory
        implements ResourceLoaderAware {

    // SPI name
    public static final String NAME = "lemmagen";

    // Configuration key
    private static final String LEXICON_KEY = "lexicon";
    private Lemmatizer lemmatizer = null;
    private final String lexiconPath;

    /** Creates a new LemmagenTokenFilterFactory */
    public LemmagenTokenFilterFactory(Map<String,String> args) {
        super(args);
        lexiconPath = require(args, LEXICON_KEY);
        if (!args.isEmpty()) {
            throw new IllegalArgumentException("Unknown parameters: " + args);
        }
    }

    private Lemmatizer createLemmatizer(InputStream lexiconInputStream) {
        try {
            return LemmatizerFactory.read(lexiconInputStream);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void inform(ResourceLoader loader) throws IOException {
        this.lemmatizer = createLemmatizer(loader.openResource(lexiconPath));
    }

    public LemmagenTokenFilterFactory() {
        throw defaultCtorException();
    }

    public TokenStream create(TokenStream input) {
        return new LemmagenTokenFilter(input, lemmatizer);
    }
}
