package ai.vespa.linguistics.pl;

import com.yahoo.container.di.componentgraph.Provider;
import org.apache.lucene.analysis.Analyzer;

public class PolishAnalyzer implements Provider<Analyzer> {
    @Override
    public Analyzer get() {
        System.err.println("Hooray - we are using the plugin to provide a polish analyzer");
        return new org.apache.lucene.analysis.pl.PolishAnalyzer();
    }

    @Override
    public void deconstruct() {}
}
