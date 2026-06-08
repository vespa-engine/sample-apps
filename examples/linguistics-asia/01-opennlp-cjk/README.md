<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

# 01 — OpenNLP with CJK segmentation

The stock Vespa container ships with `OpenNlpLinguistics`. CJK handling is *off* by default. Turn it on with two config flags and you get character-bigram-based segmentation for Chinese, Japanese, Korean — zero Java, no Maven build, just XML.

This is the smallest possible Vespa app that handles Chinese at all. Good starting point if you have a small zh-CN product catalog and don't need word-level segmentation quality.

## What this app does

- `services.xml` declares `OpenNlpLinguistics` with `<cjk>true</cjk>` and `<createCjkGrams>true</createCjkGrams>`.
- Indexed Chinese text is split into overlapping 2-character grams. "无线蓝牙耳机" → `无线`, `线蓝`, `蓝牙`, `牙耳`, `耳机`. Query terms get the same treatment, so BM25 works.
- Latin text continues to tokenize the way OpenNLP normally does.

## Limits

- **No script-aware segmentation.** Simplified and Traditional are tokenized the same way (bigrams over whatever code points show up). A zh-CN query for `手机壳` will *not* match a zh-TW doc containing `手機殼` — different code points, different bigrams.
- **No stemming, no stopwords, no synonyms.** This is positional segmentation only.
- **Index size grows.** Bigrams produce roughly N postings per N-character field instead of word-count postings.

If any of those bite you, jump to [02-lucene-per-variant](../02-lucene-per-variant) (word segmentation), [04-jieba](../04-jieba) (Jieba dictionary segmentation), or [06-opennlp-icu](../06-opennlp-icu) (cross-variant recall).

## Deploy to Vespa Cloud

Sign up for a free Vespa Cloud trial at <https://cloud.vespa.ai/>, then:

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-01     # replace TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

# Query — Simplified term should hit the Simplified doc only
vespa query 'yql=select * from product where default contains "手机壳"' \
            'model.locale=zh-CN'

# Same product in Traditional — Simplified query does NOT match it here
vespa query 'yql=select * from product where default contains "手機殼"' \
            'model.locale=zh-TW'
```

See [../ext/queries.md](../ext/queries.md) for the full set of compare-queries used across all sub-examples.

## Inspecting the tokens

The schema declares a `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) exposing per-field token streams. Query it and read `title_tokens` / `body_tokens` in the response:

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

For a Simplified doc you'll see 2-char gram tokens like `手机`, `机壳`; for Traditional, `手機`, `機殼` — different code points, different bigrams, no cross-variant hits.

## Local deploy + test

The local Docker flow doubles as the CI smoke test. The Vespa sample-apps CI parses the `data-test` attributes below: `<pre data-test="exec">` blocks run as shell, and `data-test-assert-contains="STR"` requires `STR` in the captured stdout. The Full conventions in [parent README — Testing method](../README.md#testing-method).

Run from this sub-example directory:

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/01-opennlp-cjk
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 vespaengine/vespa
vespa config set target local
vespa status deploy --wait 300
vespa deploy --wait 300 ./app
</pre>

<pre data-test="exec">
vespa feed ../ext/documents.jsonl
</pre>

<pre data-test="exec" data-test-assert-contains="cn-001">
vespa query 'yql=select id from product where default contains "手机壳"' \
            'model.locale=zh-CN'
</pre>

<pre data-test="exec" data-test-assert-contains="tw-001">
vespa query 'yql=select id from product where default contains "手機殼"' \
            'model.locale=zh-TW'
</pre>

<pre data-test="exec" data-test-assert-contains="手机">
vespa query 'yql=select * from product where id contains "cn-001"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## References

- [Vespa OpenNLP linguistics](https://docs.vespa.ai/en/linguistics/linguistics-opennlp.html)
- [Vespa newsletter — CJK support in OpenNLP](https://blog.vespa.ai/vespa-newsletter-august-2024/)
