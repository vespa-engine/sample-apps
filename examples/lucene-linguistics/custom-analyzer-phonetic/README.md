<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample applications - Lucene Linguistics

This demonstrates how to use a custom analyzer in [Lucene Linguistics](https://docs.vespa.ai/en/linguistics/lucene-linguistics.html), when the class you need doesn't come with the default linguistics bundle. One such example is the [Phonetic Token Filter](https://github.com/apache/lucene/blob/releases/lucene/9.11.1/lucene/analysis/phonetic/src/java/org/apache/lucene/analysis/phonetic/PhoneticFilterFactory.java).

## TL;DR:
* write the analyzer definition in `services.xml`
* use the analyzer in the schema
* add the phonetic library to the dependencies
* compile the application package
* deploy the application
* feed a sample document
* query

## Analyzer definition

Our sample definition in `services.xml` is:
```xml
          <analysis>
            <item key="profile=soundex">
              <tokenizer>
                <name>standard</name>
              </tokenizer>
              <tokenFilters>
                <item>
                  <name>lowercase</name>
                </item>
                <item>
                  <name>phonetic</name>
                  <conf>
                    <item key="encoder">soundex</item>
                    <item key="inject">false</item>
                  </conf>
                </item>
              </tokenFilters>
            </item>
          </analysis>
```

Where:
* `profile=soundex` is the name of the [linguistics profile](https://docs.vespa.ai/en/linguistics/linguistics.html#linguistics-profiles). We'll use it in the schema. Notice how the ID only includes the profile name, not the language. This means it applies to all languages (unless otherwise specified).
* We use the `standard` tokenizer and `lowercase` token filter.
* We use the `phonetic` token filter with the `soundex` encoder (which does the phonetic encoding). `inject=false` means we only keep the encoded tokens, not the original tokens.

## Schema

We reference the `soundex` profile in the schema for our sample field `my_text`:
```
        field my_text type string {
            indexing: summary | index
            index: enable-bm25
            linguistics {
                profile: soundex
            }
        }
```

## Application package dependency

Phonetic analyzers don't come with the default linguistics bundle, so we need to add this:

```xml
    <dependency>
      <groupId>org.apache.lucene</groupId>
      <artifactId>lucene-analysis-phonetic</artifactId>
      <version>${lucene.version}</version>
    </dependency>
```

## Compile the application package

```bash
mvn clean package
```

## Deploy the application

```bash
vespa deploy target/application
```

## Feed a sample document

```http
POST /document/v1/test/doc/docid/1
{
  "fields": {
		"my_text": "hello world"
	}
}
```

## Query

```http
POST /search/
{
  "yql": "select * from doc where my_text contains text('hela')",
  "presentation.summary": "debug-text-tokens",
  "model.locale": "en",
  "trace.level": 2
}
```

Notice that:
* We used the [text operator](https://docs.vespa.ai/en/reference/querying/yql.html#text) to only apply our defined linguistics profile to the query text `hela`.
* In the query trace, we can see the phonetic encoding:
```
YQL query parsed: [select * from doc where weakAnd(my_text contains ({stem: false, normalizeCase: false, accentDrop: false, implicitTransforms: false}\"H400\"))]
```
* The same `H400` results from the token `hello` (which sounds similar to `hela`) and this is why the document matches:
```
			{
				"id": "id:test:doc::1",
				"relevance": 0.38186238359951247,
				"source": "content",
				"fields": {
					"sddocname": "doc",
					"documentid": "id:test:doc::1",
					"my_text": "hello world",
					"text_tokens": [
						"H400",
						"W643"
					]
				}
			}
```