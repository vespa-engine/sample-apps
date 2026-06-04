<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

# 08 — Lucene phrase dictionary (custom SPI factory)

A custom Lucene `TokenFilterFactory` plus the matching `TokenFilter`, registered through `META-INF/services/` so `services.xml` can use the name `phraseDict` like any built-in filter. The filter reads a phrase dictionary from `configDir` at startup and **aggregates token streams back into single phrase tokens** whenever a consecutive run of tokens matches a dictionary entry.

This solves a concrete problem in the ICU CJK pipeline: `icuTokenizer` segments `空气净化器` into multiple tokens (something like `[空气, 净化, 器]`). Downstream filters that depend on whole-phrase identity — synonym lookups, keyword-marker protection, multi-token exact-match — never see the phrase as a single unit. With a dictionary entry `空气净化器`, this filter combines those tokens back into one before the next filter runs.

## When to choose 08

| Need | Pick |
|---|---|
| Generic ICU trad↔simp, config-only | [03-lucene-icu](../03-lucene-icu) |
| ICU + a real Chinese phrase dictionary (CKIP / HanLP / IK / Jieba user-dict / brand list / catalogue terms) | **08 (this one)** |
| Same outcome via Java instead of config | [06-opennlp-icu](../06-opennlp-icu) |
| Region-aware (s2hk, s2twp, …) | [07-opennlp-opencc](../07-opennlp-opencc) |

08 is the **template for plugging any text-based phrase dictionary** into a Lucene analyzer chain via SPI. Replace `dictionary.txt` with the output of Academia Sinica CKIP, HanLP, IK, or any in-house brand/catalogue dictionary — same shape.

## The two classes

[`PhraseDictTokenFilter.java`](src/main/java/ai/vespa/examples/linguistics/asia/lucene/PhraseDictTokenFilter.java) — extends `TokenFilter`, marked `final` (Lucene's contract). Buffers the input stream then does greedy longest-match phrase scanning, capped by the longest entry observed at dictionary load time.

[`PhraseDictTokenFilterFactory.java`](src/main/java/ai/vespa/examples/linguistics/asia/lucene/PhraseDictTokenFilterFactory.java) — extends `TokenFilterFactory` and implements `ResourceLoaderAware` to read the dictionary file from `configDir` via Vespa's resource loader.

SPI registration in [`META-INF/services/org.apache.lucene.analysis.TokenFilterFactory`](src/main/resources/META-INF/services/org.apache.lucene.analysis.TokenFilterFactory):

```
ai.vespa.examples.linguistics.asia.lucene.PhraseDictTokenFilterFactory
```

The class's static `NAME` is `phraseDict` — that's the name used in `services.xml`.

## Wiring in services.xml

```xml
<config name="com.yahoo.language.lucene.lucene-analysis">
  <configDir>linguistics</configDir>
  <analysis>
    <item key="zh-CN">
      <tokenizer><name>icu</name></tokenizer>
      <tokenFilters>
        <item>
          <name>icuTransform</name>
          <conf><item key="id">Traditional-Simplified</item></conf>
        </item>
        <item>
          <name>phraseDict</name>
          <conf><item key="dictionary">dictionary.txt</item></conf>
        </item>
        <item><name>lowercase</name></item>
      </tokenFilters>
    </item>
    <!-- zh-TW: same chain -->
  </analysis>
</config>
```

Filter order matters. The aggregator runs **after** `icuTransform` so dictionary entries can be written in Simplified — they match the post-fold token stream regardless of whether the source document is zh-CN or zh-TW.

## The dictionary file

[`linguistics/dictionary.txt`](src/main/application/linguistics/dictionary.txt) — one phrase per line, comments with `#`. In this sample:

```
空气净化器
空气清净机
笔记本电脑
笔记型电脑
智能手表
智慧型手表
电饭煲
电子锅
保温杯
运动跑鞋
```

Swap in your real Chinese dictionary by replacing this file. Common sources:

- **Academia Sinica CKIP** (中央研究院 CKIP 中文斷詞系統) — export the dictionary as plain text, one term per line.
- **HanLP** — `data/dictionary/*.txt` lexicon files.
- **IK Analyzer** — `main2012.dic` style.
- **Jieba user-dict** — drop the third (frequency/POS) column.
- **Your in-house catalogue** — product names, brand variants, model numbers.

The filter is format-agnostic: any text file with one phrase per line (post-fold form) will work.

## Build, test, deploy to Vespa Cloud

Sign up at <https://cloud.vespa-cloud.com/>, then:

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-08     # replace TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# Multi-character phrase queries hit clean (single-token) matches.
vespa query 'yql=select * from product where default contains "空气净化器"' 'model.locale=zh-CN'
# -> cn-009 (phrase indexed and queried as a single token)

vespa query 'yql=select * from product where default contains "空氣清淨機"' 'model.locale=zh-TW'
# -> tw-009 (ICU fold to 空气清净机 then aggregator combines to single token)

# Cross-variant via ICU still works
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
# -> cn-001 and tw-001
```

## Inspecting the tokens

Use the `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) to confirm the aggregator collapsed multi-character phrases into single tokens:

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

Expect `title_tokens` arrays like `[..., "空气净化器", ...]` and `[..., "笔记本电脑", ...]` — each dictionary phrase as a single element, not split into per-character bigrams or word fragments.

## Local deploy + test

The local Docker flow doubles as the CI smoke test. The Vespa sample-apps CI parses the `data-test` attributes below: `<pre data-test="exec">` blocks run as shell, and `data-test-assert-contains="STR"` requires `STR` in the captured stdout. The Full conventions in [parent README — Testing method](../README.md#testing-method).

Run from this sub-example directory:

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/08-lucene-phrase-dict
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

<pre data-test="exec" data-test-assert-contains="空气净化器">
vespa query 'yql=select * from product where id contains "cn-009"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## Combining with synonyms

To bridge regional vocabulary (`空气净化器` ↔ `空气清净机`, `笔记本电脑` ↔ `笔记型电脑`), pair this aggregator with the synonym filter pattern shown in [03-lucene-icu](../03-lucene-icu). The aggregator gives the synonym filter clean single-token inputs to match against. Full chain shape:

```xml
<tokenFilters>
  <item><name>icuTransform</name>...</item>
  <item><name>phraseDict</name>...</item>      <!-- 08 -->
  <item><name>synonymGraph</name>...</item>     <!-- 03 -->
  <item><name>flattenGraph</name></item>
  <item><name>lowercase</name></item>
</tokenFilters>
```

Behaviour in practice depends on how Lucene's synonym filters interact with the graph alternatives — see 03's notes for tuning guidance.

## References

- [Lucene `TokenFilterFactory` SPI](https://lucene.apache.org/core/9_8_0/analysis/common/org/apache/lucene/analysis/TokenFilterFactory.html)
- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- Sibling: [`examples/lucene-linguistics/add-token-filter-factory`](../../lucene-linguistics/add-token-filter-factory) — generic Lucene SPI factory template (non-Chinese specific)
- [Academia Sinica CKIP](https://ckip.iis.sinica.edu.tw/) | [HanLP](https://github.com/hankcs/HanLP) | [IK Analyzer](https://github.com/medcl/elasticsearch-analysis-ik) | [Jieba user-dict format](https://github.com/fxsjy/jieba#%E8%BD%BD%E5%85%A5%E8%AF%8D%E5%85%B8)
