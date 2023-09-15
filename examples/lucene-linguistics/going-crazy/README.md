# Vespa Lucene Linguistics: Going Crazy

## TL;DR

Search problems get really complicated when you need to deal with multilingual aspects.
Lucene has a battle-tested and standards compliant set of available libraries to help you solve your problems.

## Context

The goals of this application package are:
- set up OpenNLP tokenizers;
- set up Lemmagen token filters with sample resource files; 
- construct an analyzer entirely in Java code and register it as a component;

## Analysis components

Lucene has plenty of components [available](https://lucene.apache.org/core/9_7_0/index.html).
One of which is [`analysis-opennlp`](https://lucene.apache.org/core/9_7_0/analysis/opennlp/index.html).

### OpenNLP

The OpenNLP library adds 1 tokenizer identified with `openNlp`, and 3 token filters:
`openNlpLemmatizer`, `openNlpChunker`, `openNlppos`.  

Let's set a `org.apache.lucene.analysis.opennlp.OpenNLPTokenizerFactory` and
`org.apache.lucene.analysis.snowball.SnowballPorterFilterFactory`.

### Feed Documents

```shell
vespa feed src/main/application/ext/documents/*
```
