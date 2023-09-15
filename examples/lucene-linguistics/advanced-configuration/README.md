# Vespa Lucene Linguistics

This Vespa application package (VAP) previews the configuration options of the `lucene-linguistics` package.
Probably the main benefit of the `LuceneLinguistics` is the configurability when compared to other `Linguistics` implementations.

## Custom Lucene Analyzers

There are multiple ways to use a Lucene `Analyzer` for a language.
Each analyzer is identified by a language key, e.g. 'en' for English language. 
These are Analyzer types in the order of descending priority:
1. Created through the `Linguistics` component configuration.
2. An `Analyzer` wrapped into a Vespa `<component>`.
3. A list of [default Analyzers](https://github.com/vespa-engine/vespa/blob/5d26801bc63c35705e708d3cc7086f0b0103e909/lucene-linguistics/src/main/java/com/yahoo/language/lucene/DefaultAnalyzers.java) per language.
4. The `StandardAnalyzer`.

### Add a Lucene Analyzer component

Vespa provides a `ComponentRegistry` mechanism.
The `LuceneLinguistics` accepts a `ComponentRegistry<Analyzer>` into the constructor.
Basically, the Vespa container at start time collects all the components that are of the `Analyzer` type automagically.

To declare such components:
```xml
<component id="en"
           class="org.apache.lucene.analysis.core.SimpleAnalyzer"
           bundle="vespa-lucene-linguistics-poc" />
```
Where:
- `id` should contain a language code.
- `class` should be the implementing class.
Note that it is a class straight from the Lucene library.
Also, you can create an `Analyzer` class just inside your VAP and refer it.
- `bundle` must be your application package `artifactId` as specified in the `pom.xml`.

Here are two types of `Analyzer` components:
1. That doesn't require any setup.
2. That requires a setup (e.g. constructor with arguments).

The previous component declaration example is of type (1).

The (2) type requires a bit more work.

Create a class (e.g. for the Polish language):
```java
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
```

Add a component declaration into the `services.xml` file:
```xml
<component id="pl"
           class="ai.vespa.linguistics.pl.PolishAnalyzer"
           bundle="vespa-lucene-linguistics-poc" />
```
And now you have the handling of the Polish language available.
