package ai.vespa.linguistics.lemmagen;

import eu.hlavki.text.lemmagen.api.Lemmatizer;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;

import java.io.IOException;

/**
 * Code is loosely based on
 * https://github.com/vhyza/elasticsearch-analysis-lemmagen/blob/master/src/main/java/org/elasticsearch/index/analysis/LemmagenFilter.java
 */
public final class LemmagenTokenFilter extends TokenFilter {

    private final CharTermAttribute termAttr = addAttribute(CharTermAttribute.class);
    private final KeywordAttribute keywordAttr = addAttribute(KeywordAttribute.class);
    private final Lemmatizer lemmatizer;

    public LemmagenTokenFilter(final TokenStream input, final Lemmatizer lemmatizer) {
        super(input);
        this.lemmatizer = lemmatizer;
        System.err.println("Hooray - we are using the LemmagenTokenFilter plugin");
    }

    public boolean incrementToken() throws IOException {
        if (!input.incrementToken()) {
            return false;
        }
        CharSequence lemma = lemmatizer.lemmatize(termAttr);
        if (!keywordAttr.isKeyword() && !equalCharSequences(lemma, termAttr)) {
            termAttr.setEmpty().append(lemma);
        }
        return true;
    }

    private boolean equalCharSequences(CharSequence s1, CharSequence s2) {
        int len1 = s1.length();
        int len2 = s2.length();
        if (len1 != len2)
            return false;
        for (int i = len1; --i >= 0;) {
            if (s1.charAt(i) != s2.charAt(i)) {
                return false;
            }
        }
        return true;
    }
}
