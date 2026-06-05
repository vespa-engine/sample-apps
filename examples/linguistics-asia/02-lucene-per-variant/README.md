<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

# 02 — Lucene linguistics, one analyzer per Chinese variant

`LuceneLinguistics` lets you compose any Apache Lucene analyzer chain per language code. This sub-example shows the **per-variant** wiring: `zh-CN` documents and queries go through SmartCN (HMM word segmenter trained on Simplified); `zh-TW` documents and queries go through Standard tokenizer + CJK bigram filter. They live in separate indices of the same field, isolated.

This is the option to pick when you want **proper word segmentation for one variant** and **independent variant treatment** — i.e. zh-CN and zh-TW are different markets with different storefronts and you don't want them to bleed into each other's results.

Need cross-variant recall (zh-CN query finds zh-TW docs)? Use [03-lucene-icu](../03-lucene-icu) — same Lucene chain plus an OpenCC `CharFilter` that folds Traditional to Simplified before tokenizing.

## What this app does

`services.xml` declares `LuceneLinguistics` with three entries in the analysis map:

```xml
<item key="zh-CN">
  <tokenizer><name>hmmChinese</name></tokenizer>
  <tokenFilters>
    <item><name>lowercase</name></item>
  </tokenFilters>
</item>
<item key="zh-TW">
  <tokenizer><name>standard</name></tokenizer>
  <tokenFilters>
    <item><name>cjkWidth</name></item>
    <item><name>lowercase</name></item>
    <item><name>cjkBigram</name></item>
  </tokenFilters>
</item>
```

- **zh-CN** → SmartCN's `hmmChinese` tokenizer outputs word-level tokens for Simplified text: `无线蓝牙耳机` → `无线`, `蓝牙`, `耳机`. Stop-word filtering is omitted — there is no `smartChineseStop` SPI in Lucene; if you want stopword removal, use a generic `stop` filter with your own stopword list.
- **zh-TW** → SmartCN is trained on Simplified and segments Traditional poorly. We instead split with `standard` (codepoint-based) then expand to overlapping bigrams (Lucene's `cjkBigram` filter, the same approach as Solr/Elasticsearch for Traditional).
- **No `zh` fallback entry** — `zh` and `zh-CN` both resolve to `Language.CHINESE_SIMPLIFIED` in Vespa's enum, so listing both as separate `<item>` entries causes the later one to overwrite the earlier. Use only the specific variant keys.

## Build and deploy to Vespa Cloud

Sign up for a free Vespa Cloud trial at <https://cloud.vespa-cloud.com/>, then:

```sh
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-02     # replace TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# zh-CN query, word-segmented by SmartCN
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' \
            'model.locale=zh-CN' 'trace.level=2' \
  | jq '.trace.children | last | .children[] | select(.message) | select(.message | test("YQL.*")) | .message'

# zh-TW query, bigrammed
vespa query 'yql=select * from product where default contains "無線藍牙耳機"' \
            'model.locale=zh-TW'
```

Use `trace.level=2` to see how each analyzer chain rewrites the query. SmartCN should produce three words; the zh-TW chain produces overlapping bigrams.

## Inspecting the tokens

The schema declares a `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) — query it to read per-field token streams without shelling into the container:

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

Expect zh-CN `title_tokens` to contain word-level segments (`无线`, `蓝牙`, `耳机`); zh-TW `title_tokens` to contain bigrams (`無線`, `線藍`, `藍牙`, …).

## Local deploy + test

The local Docker flow doubles as the CI smoke test. The Vespa sample-apps CI parses the `data-test` attributes below: `<pre data-test="exec">` blocks run as shell, and `data-test-assert-contains="STR"` requires `STR` in the captured stdout. The Full conventions in [parent README — Testing method](../README.md#testing-method).

Run from this sub-example directory:

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/02-lucene-per-variant
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 vespaengine/vespa
vespa config set target local
vespa status deploy --wait 300
mvn -q clean package
vespa deploy --wait 300 target/application
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

<pre data-test="exec" data-test-assert-contains="无线">
vespa query 'yql=select * from product where id contains "cn-002"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## Why SmartCN is not safe to use on Traditional

SmartCN's HMM was trained on a Simplified corpus. Hand it Traditional input and segmentation boundaries degrade — you get plausible-looking words that don't actually match the documents users wrote. CJK bigram is dumber but consistent. If you want SmartCN-quality segmentation on Traditional, the realistic path is to fold to Simplified first; see [03-lucene-icu](../03-lucene-icu).

## References

- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- [Lucene `smartcn` analyzer](https://lucene.apache.org/core/9_8_0/analysis/smartcn/index.html)
- [Lucene `cjkBigram` filter](https://lucene.apache.org/core/9_8_0/analysis/common/org/apache/lucene/analysis/cjk/CJKBigramFilter.html)
