<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Lucene Linguistics 

This app demonstrates how to use multiple analyzer profiles in [Lucene Linguistics](https://docs.vespa.ai/en/linguistics/lucene-linguistics.html).

You can bind different fields to different analyzer profiles in the schema. Here, we have three analyzers in [services.xml](app/services.xml):
- `lowerFolding`: [standard tokenizer](https://lucene.apache.org/core/9_11_1/core/org/apache/lucene/analysis/standard/StandardTokenizer.html) + [lowercase](https://lucene.apache.org/core/9_11_1/core/org/apache/lucene/analysis/LowerCaseFilter.html) and [ASCII folding](https://lucene.apache.org/core/9_11_1/analysis/common/org/apache/lucene/analysis/miscellaneous/ASCIIFoldingFilterFactory.html) token filters
- `lowerFoldingStemming`: lowerFolding + [kStem for English](https://lucene.apache.org/core/9_11_1/analysis/common/org/apache/lucene/analysis/en/KStemFilterFactory.html)
- `lowerFoldingStemmingSynonyms`: lowerFoldingStemming + [synonym expansion](https://lucene.apache.org/core/9_11_1/analysis/common/org/apache/lucene/analysis/synonym/SynonymGraphFilterFactory.html)

We have three fields in the schema:
- `title`: bound to `lowerFolding`
- `description`: bound to `lowerFoldingStemming` at write time, and `lowerFoldingStemmingSynonyms` at search time. We want to expand synonyms at search time only, it doesn't make sense to do it on both sides.

In this example, we only use English, but you can combine this with multiple languages if you wanted to. Steps to do this are:
1. In `services.xml`, define an analyzer for each profile+language combination.
   - Use `default` profile for fields that are not bound to a specific profile.
2. In the schema, use `linguistics` block to bind the field to the profile (or profiles, if you need different profiles for index and search).
3. Use [language tags and detection](https://docs.vespa.ai/en/linguistics/linguistics.html#language-handling) as before.

## Deploy the application
Follow [app deploy guide](https://docs.vespa.ai/en/basics/deploy-an-application)
through the <code>vespa deploy</code> step, cloning `examples/lucene-linguistics/multiple-profiles` instead of `album-recommendation`.

## Feed the sample document

```bash
vespa feed ext/*.json
```

## Run queries

### Basic example

This will confirm that ASCII folding is working on the `title` field, because it will match `åao` with `åäö`:
```bash
curl -s -X POST -d '{
  "yql":"select * from sources * where title contains \"åao\"",
  "presentation.summary": "debug-text-tokens",
  "model.locale": "en",
  "trace.level":2}' -H "Content-Type: application/json" 'http://localhost:8080/search/' | jq .
```

### Different profiles for query and index

For the `description` field, [the schema defines a different profile for search time](app/schemas/doc.sd) which does synonym expansion. So it will match "dubious" from the query string with "special" in the index:

```bash
curl -s -X POST -d '{
  "yql":"select * from sources * where description contains \"dubious\"",
  "presentation.summary": "debug-text-tokens",
  "model.locale": "en",
  "trace.level":2}' -H "Content-Type: application/json" 'http://localhost:8080/search/' | jq .
```

### Force a different profile for the query

`model.type.profile` defines the profile to use for parsing the query string. This will match "dubious" with "special" (our test synonym expansion) even for the `title` field (which is bound to the `lowerFolding` profile which doesn't do synonym expansion):

```bash
curl -s -X POST -d '{
  "yql":"select * from sources * where title contains \"dubious\"",
  "model.type.profile": "lowerFoldingStemmingSynonyms",
  "presentation.summary": "debug-text-tokens",
  "model.locale": "en",
  "trace.level":2}' -H "Content-Type: application/json" 'http://localhost:8080/search/' | jq .
```

### Force a different profile for a specific query clause

This works with `userInput()` and the `grammar.profile` annotation:

```bash
curl -s -X POST -d '{
  "yql":"select * from sources * where where {defaultIndex:'title', grammar.profile: 'lowerFoldingStemmingSynonyms', grammar:'linguistics'}userInput('dubious')",
  "presentation.summary": "debug-text-tokens",
  "model.locale": "en",
  "trace.level":2}' -H "Content-Type: application/json" 'http://localhost:8080/search/' | jq .
```

**NOTE**: The `grammar: 'linguistics'` annotation isn't required in this case, but makes sure that no additional parsing (besides the defined profile) is done. This is useful, for example, with collapsing synonyms (e.g., `wi fi => wifi`). Otherwise, the query becomes `["wi", "fi"]` along the way.