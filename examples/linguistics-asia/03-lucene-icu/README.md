<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

# 03 — Lucene linguistics with ICU cross-variant folding

The cross-variant cousin of [02-lucene-per-variant](../02-lucene-per-variant). Same `LuceneLinguistics` plumbing, but the analyzer chain runs:

1. **`icuTokenizer`** — Unicode-aware CJK word segmentation (replaces SmartCN; dictionary ships with the Lucene `lucene-analysis-icu` JAR).
2. **`icuTransform` with `id=Traditional-Simplified`** — ICU's standard Traditional → Simplified transliterator. Applied uniformly on index and query sides, so a zh-CN query matches a zh-TW document and vice versa.
3. `lowercase` for the Latin tokens that come along for the ride.

No custom Java. No third-party library beyond what Lucene already ships.

## When to use this vs 02

| | [02-lucene-per-variant](../02-lucene-per-variant) | **03 — this one** |
|---|---|---|
| zh-CN ↔ zh-TW recall | isolated (independent storefronts) | unified (one storefront serving both markets) |
| Tokenizer | SmartCN for zh-CN, Standard+cjkBigram for zh-TW | ICU CJK word seg for both |
| Cross-variant fold | none | yes (ICU transliterator) |
| Java code | none | none |
| Build | Maven (smartcn dep) | Maven (icu dep) |

## How the chain plugs in

`services.xml`:

```xml
<item key="zh-CN">
  <tokenizer><name>icu</name></tokenizer>
  <tokenFilters>
    <item>
      <name>icuTransform</name>
      <conf>
        <item key="id">Traditional-Simplified</item>
      </conf>
    </item>
    <item><name>lowercase</name></item>
  </tokenFilters>
</item>
```

The `zh-TW` entry is identical — the whole point is variant labels stop mattering once the transliteration step has run.

The `<conf><item key="id">Traditional-Simplified</item></conf>` block is how lucene-linguistics passes SPI factory parameters; this is the same `<conf>` shape used elsewhere in the [lucene-linguistics samples](../../lucene-linguistics/going-crazy).

## Build and deploy to Vespa Cloud

Sign up at <https://cloud.vespa.ai/>, then:

```sh
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-03     # replace TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# Cross-variant queries — both should hit cn-001 AND tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'

# CJK word-level segmentation — should hit both cn-002 (无线蓝牙耳机) and tw-002 (無線藍牙耳機)
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' 'model.locale=zh-CN'
```

## Inspecting the tokens

Use the `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) to confirm the trad→simp fold actually landed in the index:

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

Expect every doc's `title_tokens` / `body_tokens` to contain Simplified characters only — Traditional source text from the zh-TW documents should appear under `title` / `body` summary fields but not in the `_tokens` arrays.

## Local deploy + test

The local Docker flow doubles as the CI smoke test. The Vespa sample-apps CI parses the `data-test` attributes below: `<pre data-test="exec">` blocks run as shell, and `data-test-assert-contains="STR"` requires `STR` in the captured stdout. The Full conventions in [parent README — Testing method](../README.md#testing-method).

Run from this sub-example directory:

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/03-lucene-icu
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

<pre data-test="exec" data-test-assert-contains="tw-001">
vespa query 'yql=select id from product where default contains "手机壳"' \
            'model.locale=zh-CN'
</pre>

<pre data-test="exec" data-test-assert-contains="cn-001">
vespa query 'yql=select id from product where default contains "手機殼"' \
            'model.locale=zh-TW'
</pre>

<pre data-test="exec" data-test-assert-contains="无线">
vespa query 'yql=select * from product where id contains "tw-002"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## Closing lexical gaps with synonyms

Transliteration handles script-level differences (`手機殼` ↔ `手机壳`). It does NOT handle regional vocabulary differences — a zh-CN doc `空气净化器` and a zh-TW doc `空氣清淨機` *both fold to Simplified* but the words remain `空气净化器` vs `空气清净机`. To close that gap, the chain runs a Lucene `synonymGraph` filter (with `flattenGraph` so the alternatives stay in the index):

```xml
<tokenFilters>
  <item>
    <name>icuTransform</name>
    <conf><item key="id">Traditional-Simplified</item></conf>
  </item>
  <item>
    <name>synonymGraph</name>
    <conf>
      <item key="synonyms">synonyms.txt</item>
      <item key="expand">true</item>
      <item key="tokenizerFactory">org.apache.lucene.analysis.icu.segmentation.ICUTokenizerFactory</item>
    </conf>
  </item>
  <item><name>flattenGraph</name></item>
  <item><name>lowercase</name></item>
</tokenFilters>
```

[`linguistics/synonyms.txt`](src/main/application/linguistics/synonyms.txt) — Solr synonym format, comma-separated equivalent groups. Keep the source-of-truth list **all-Simplified** because the transliteration step has already normalized to Simplified by the time the synonym filter runs.

`configDir` (declared at the top of the `<config>` block) tells `LuceneLinguistics` where to find resource files relative to the application package root.

### Verified

```sh
vespa query 'yql=select * from product where default contains "锅"' 'model.locale=zh-CN'
# → hits BOTH 电饭煲 (cn-008) and 电子鍋 (tw-008) — single-char synonym 锅 ⇔ 煲 bridges the lexical gap.
```

### Multi-character CJK caveats

`tokenizerFactory=…ICUTokenizerFactory` makes the synonym file parse with the same ICU CJK rules as your runtime text, so multi-character entries like `空气净化器,空气清净机` *should* match. In practice ICU's word-boundary heuristics can split your synonym entry into slightly different token sequences than the documents and queries you index, and the synonym filter then never finds a match. If a multi-char phrase doesn't bridge as expected:

1. `vespa query ... summary=debug-tokens` — see how runtime text actually segmented (the `title_tokens` / `body_tokens` arrays).
2. `trace.level=2` on a query — see how the query side segmented.
3. Adjust the synonym file: rewrite the entry with explicit whitespace to force token boundaries (e.g. `空 气 净 化 器, 空 气 清 净 机`), or fall back to single-character pairs for the cases that matter.

Production deployments usually maintain this file as a tuned artifact iterated against real query logs.

## Caveats

- ICU's `Traditional-Simplified` transliterator is one-way (collapses Traditional onto Simplified). The original `summary` field still renders the document's source script — only the index is normalized.
- Hong Kong orthography (zh-HK) is close to but not identical to zh-TW; ICU's `Traditional-Simplified` covers the bulk of it. If you need HK-specific behavior, ICU also ships `s2hk` and `t2hk` transliterators that can chain.
- ICU's tokenizer uses dictionary-based word segmentation. Quality is close to SmartCN for Simplified and substantially better than `cjkBigram` for Traditional.

## References

- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- [Lucene `lucene-analysis-icu`](https://lucene.apache.org/core/9_8_0/analysis/icu/index.html)
- [ICU Transliterator IDs](https://unicode-org.github.io/icu/userguide/transforms/general/#transliterator-identifiers)
