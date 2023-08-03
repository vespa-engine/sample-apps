package ai.vespa.linguistics.pl;

import com.yahoo.container.di.componentgraph.Provider;
import org.apache.lucene.analysis.Analyzer;

public class PolishAnalyzer implements Provider<Analyzer> {
    @Override
    public Analyzer get() {
        return new org.apache.lucene.analysis.pl.PolishAnalyzer();
    }

    @Override
    public void deconstruct() {}
}
