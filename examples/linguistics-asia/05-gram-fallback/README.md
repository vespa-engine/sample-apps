<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

# 05 — Gram-only fallback, no linguistics module

Sometimes the right answer is: don't try to be smart about tokenization at all. Vespa's schema language has a `match: gram` mode that splits a field into overlapping fixed-size character n-grams at index *and* query time. No `Linguistics` component required, no dictionary, no segmentation model. Works for any CJK input — and any Unicode input — because it just counts characters.

Pick this when:
- You want **maximum recall** and don't mind some over-matching.
- You can't or won't ship a Java bundle.
- You're indexing a mix of variants and scripts (zh-CN, zh-TW, Japanese kanji, etc.) and a single set of analyzer rules can't reasonably cover them all.

## What this app does

`services.xml` declares no `Linguistics` component. Vespa's stock OpenNLP linguistics is still loaded, but the gram-matched fields ignore it: they're tokenized purely by the schema's match rule.

The relevant bit of `app/schemas/product.sd`:

```
field title type string {
  indexing: index | summary
  index: enable-bm25
  match {
    gram
    gram-size: 2
  }
}
```

"无线蓝牙耳机" gets indexed as the bigrams `无线`, `线蓝`, `蓝牙`, `牙耳`, `耳机`. A query of `无线` retrieves any document containing that bigram anywhere. BM25 still ranks reasonably because the inverted-index math doesn't care whether a "token" is a word or a 2-gram.

## Deploy to Vespa Cloud

Sign up at <https://cloud.vespa.ai/>, then:

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-05     # replace TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

Expect: each query hits its own variant *and* potentially leaks into the other when bigrams overlap. That's the trade.

## Inspecting the tokens

The schema declares a `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens). Query it to see the actual 2-grams written to the index:

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

`title_tokens` arrays will show overlapping 2-grams (`手机`, `机壳` for Simplified; `手機`, `機殼` for Traditional).

## Local deploy + test

The local Docker flow doubles as the CI smoke test. The Vespa sample-apps CI parses the `data-test` attributes below: `<pre data-test="exec">` blocks run as shell, and `data-test-assert-contains="STR"` requires `STR` in the captured stdout. The Full conventions in [parent README — Testing method](../README.md#testing-method).

Run from this sub-example directory:

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/05-gram-fallback
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

<pre data-test="exec" data-test-assert-contains="机壳">
vespa query 'yql=select * from product where id contains "cn-001"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## Trade-offs

- **Index size grows**: roughly one posting per character per field. Acceptable on small catalogs (this example is fine on a laptop); plan for it on bigger ones.
- **No stemming, no stopwords, no synonyms**: BM25 over bigrams is a blunt instrument. For ecommerce on a small zh-CN-only catalog, it can still beat a sloppy word segmenter that mis-segments product names.
- **Variant leakage**: bigrams shared between Simplified and Traditional words *do* cross-match, but bigrams composed of *different* code points (e.g. `机` vs `機`) do not. So 05 is a partial cross-variant solution at best.
- **Latin tokens are unaffected**: the gram-size rule splits Chinese characters but leaves runs of Latin letters as whole tokens.

For tighter recall semantics, see [01-opennlp-cjk](../01-opennlp-cjk) (same idea, but you get the rest of OpenNLP's pipeline for non-CJK text), or move up to [02-lucene-per-variant](../02-lucene-per-variant) for word-level segmentation.

## References

- [Vespa text matching — gram](https://docs.vespa.ai/en/querying/text-matching.html#gram)
- [Schema reference — `match`](https://docs.vespa.ai/en/reference/schema-reference.html#match)
