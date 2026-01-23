<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Lucene Linguistics 

This app demonstrates how to use a custom analyzer in [Lucene Linguistics](https://docs.vespa.ai/en/linguistics/lucene-linguistics.html) without Java.

This is useful when default analyzers (e.g., language analyzers) do not meet your needs. For example, on a text like:
```text
c++ developer (*nix OS)
```

You'd often lose the `++` part. A [Pattern tokenizer](https://lucene.apache.org/core/9_11_1/analysis/common/org/apache/lucene/analysis/pattern/PatternTokenizer.html) can help here ([services.xml snippet](app/services.xml)):

```xml
<tokenizer>
  <name>pattern</name>
  <conf>
    <!-- Split on spaces and parentheses only -->
    <item key="pattern">\s|\(|\)</item>
  </conf>
</tokenizer>
```

For all the character filters, tokenizers and token filters available, check out the [Lucene analysis-common Javadoc](https://lucene.apache.org/core/9_11_1/analysis/common/allclasses-index.html).


## Deploy the application
Follow [app deploy guide](https://docs.vespa.ai/en/basics/deploy-an-application)
through the <code>vespa deploy</code> step, cloning `examples/lucene-linguistics/custom-analyzer-non-java` instead of `album-recommendation`.

## Feed test data
Feed the sample document: 

```bash
vespa feed ext/*.json
```

## Run a test query
```bash
curl -s -X POST -d '{
  "yql":"select * from sources * where text contains \"c++\"",
  "presentation.summary": "debug-text-tokens",
  "model.locale": "en",
  "trace.level":2}' -H "Content-Type: application/json" 'http://localhost:8080/search/' | jq .
```

You'd see the document match and its tokens:
```
"text_tokens": [
  "c++",
  "developer",
  "*nix",
  "os"
]
```