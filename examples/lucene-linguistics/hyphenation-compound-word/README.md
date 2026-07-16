<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# German compound splitting with Lucene Linguistics

This sample uses Lucene's `hyphenationCompoundWord` token filter to split
German compound words. The original word and dictionary-recognized parts are
indexed as alternatives at the same position. For example,
`Donaudampfschiff` produces `donaudampfschiff`, `donau`, `dampf`, and
`schiff`.

The analyzer is based on the configuration recommended by the
[German Decompounder project](https://github.com/uschindler/german-decompounder):
a standard tokenizer followed by lowercase, hyphenation-based decompounding,
German normalization, and German light stemming.

## Download the linguistic resources

The German Decompounder dictionary is LGPL-3.0-or-later, and the hyphenation
patterns use the LaTeX Project Public License. They are therefore downloaded
instead of being included in this Apache-2.0-licensed sample. See the
[upstream notice](https://github.com/uschindler/german-decompounder/blob/master/NOTICE.txt)
for details.

Run the downloader before building:

```bash
./download-resources.sh
```

The script makes a shallow clone of the upstream repository and copies
`dictionary-de.txt` and the Lucene-compatible `de_DR.xml` under
`src/main/application/lucene-linguistics/`, the directory configured by
`configDir` in `services.xml`.

## Build and deploy

This sample requires Java 17, Maven 3.6 or newer, Git, the Vespa CLI, and a
running Vespa deployment selected as the CLI target.

```bash
mvn clean package
vespa deploy --wait 300 target/application
vespa feed documents.jsonl
```

The deployable application is `target/application`, not
`src/main/application`, because the Maven build adds the custom component
bundle under `target/application/components`.

## Query compound parts

Query for `dampf`, which is a part of two sample compounds:

```bash
vespa query \
  'yql=select * from doc where title contains "dampf"' \
  'model.locale=de' \
  'presentation.summary=debug-text-tokens'
```

The result contains `id:test:doc::donau` (`Donaudampfschiff`) and
`id:test:doc::dampf` (`Dampfschiff`). The `title_tokens` summary shows the
full indexed word and its emitted parts. `model.locale=de` selects the German
analyzer for query text, while the documents set their indexing language with
the `language` field.

## Why this sample builds an application bundle

Lucene's hyphenation parser uses the Java XML packages `javax.xml.parsers`,
`org.xml.sax`, and `org.xml.sax.helpers`. These uses originate in an embedded
dependency, so automatic OSGi bundle analysis does not discover them. If
`LuceneLinguistics` is loaded from a bundle without those imports, Vespa's
container component graph fails to start with an error such as:

```text
NoClassDefFoundError: org/xml/sax/InputSource
```

No Vespa platform modification is needed. Like the
[custom analyzer phonetic sample](../custom-analyzer-phonetic/), this
application compiles `lucene-linguistics` into an application-owned bundle.
The bundle plugin configuration in `pom.xml` adds the exact missing imports:

```xml
<Import-Package>javax.xml.parsers,org.xml.sax,org.xml.sax.helpers</Import-Package>
```

The component declaration in `services.xml` deliberately names that bundle:

```xml
<component id="linguistics"
           class="com.yahoo.language.lucene.LuceneLinguistics"
           bundle="lucene-linguistics-hyphenation-compound-word">
```

Using `bundle="lucene-linguistics"` here would load the installed platform
bundle instead and bypass the imports added by this application.

Finally, the schema uses `stemming: multiple`. The decompounder emits the full
compound and its parts at the same token position; this setting preserves all
of those alternatives in the index. It's the same problem/solutions as with
[ngrams](https://docs.vespa.ai/en/linguistics/lucene-linguistics.html#indexing-all-stemmed-words).
