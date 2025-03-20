package ai.vespa.test;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.TokenFilterFactory;

import java.util.Map;

public class MyFilterFactory extends TokenFilterFactory {

    public static final String NAME = "myFilterFactory";

    // not actually used, but must be present:
    public MyFilterFactory() {
        throw defaultCtorException();
    }

    public MyFilterFactory(Map<String, String> config) {
        super(config);
	// probably plug in some configuration code here
        System.err.println("Constructed: " + this.getClass());
    }

    public TokenStream create(TokenStream input) {
        System.err.println("create called for: " + this.getClass());
	// actually create your TokenFilter and return that here:
        return input;
    }

}
