<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

# 07 — Custom Java Linguistics with OpenCC4j (regional-variant aware)

Same shape as [06-opennlp-icu](../06-opennlp-icu), with the trad↔simp fold backed by [OpenCC4j](https://github.com/houbb/opencc4j) instead of ICU4J. Functionally equivalent on the basic generic `Traditional ↔ Simplified` use case (this dataset). The reason to reach for this sub-example over 06 is **finer-grained Chinese regional variants** that ICU's single `Traditional-Simplified` bucket doesn't expose.

## When to choose 07 over 06 (or 03)

| Need | Pick |
|---|---|
| Generic Traditional ↔ Simplified, config-only | [03-lucene-icu](../03-lucene-icu) |
| Generic Traditional ↔ Simplified, Java escape hatch | [06-opennlp-icu](../06-opennlp-icu) |
| **Hong Kong (zh-HK) orthography distinct from Taiwan** | **07 (this one) — or a Lucene factory wrapping OpenCC4j (see note)** |
| **Region-specific phrase tables** (Taiwan-specific phrases vs Mainland) | **07** — or a Lucene factory wrapping OpenCC4j |
| **Bidirectional region-aware conversion** (s2hk, hk2s, tw2s, s2twp …) | **07** — or a Lucene factory wrapping OpenCC4j |

> **Note on the Lucene-side alternative.** OpenCC4j is a library, not an integration point. 07 wraps it in an `OpenNlpLinguistics` subclass; you can equally wrap it in a Lucene `TokenFilterFactory` by combining 08's SPI-factory pattern with 07's `ZhConverterUtil` call. Same library, Lucene config path. We don't ship that combination as a separate sub-example because it's a straightforward mechanical merge of 07 and 08, but it's the right choice if you've otherwise committed to the Lucene path and want to stay there.

OpenCC ships several directional converters that ICU does not:

| OpenCC direction | What it does | When you need it |
|---|---|---|
| `s2t` | Simplified → Traditional (generic) | Generic conversion |
| `s2tw` | Simplified → Traditional (Taiwan) | Taiwan-specific orthography |
| `s2hk` | Simplified → Traditional (Hong Kong) | Hong Kong orthography |
| `s2twp` | Simplified → Traditional Taiwan + phrases | Region term substitution |
| `t2s` | Traditional → Simplified | What this sub-example uses |
| `tw2s` | Taiwan Traditional → Simplified | Reverse of `s2tw` |
| `hk2s` | Hong Kong Traditional → Simplified | Reverse of `s2hk` |

If your catalogue mixes zh-TW *and* zh-HK content (different orthographies for the same product) and you want each direction normalized correctly, OpenCC is the cleaner choice.

If you only care about "trad → simp, doesn't matter which trad", **stay with 03 or 06** — ICU is shipped with Vespa, no extra dependency, simpler to maintain.

## The two classes

[`OpenCcLinguistics.java`](src/main/java/ai/vespa/examples/linguistics/asia/OpenCcLinguistics.java) — same structure as 06's `IcuLinguistics`. Extends `OpenNlpLinguistics`, wraps tokenizer, overrides `getTokenizer` / `getStemmer` / `getSegmenter`.

[`OpenCcTokenizer.java`](src/main/java/ai/vespa/examples/linguistics/asia/OpenCcTokenizer.java) — per-token fold using `ZhConverterUtil.toSimple(originText)`. Same `getOrig()` != `getTokenString()` pattern as 06 (see [06 README](../06-opennlp-icu/README.md#the-pattern-that-matters-per-token-tokenstring-not-pre-normalization-of-input) for the underlying Vespa behaviour this guards against).

```java
private Iterable<Token> wrap(Iterable<Token> raw) {
    List<Token> out = new ArrayList<>();
    for (Token t : raw) {
        String origText = t.getOrig();
        SimpleToken st = new SimpleToken(origText)
                .setOffset(t.getOffset())
                .setType(t.getType())
                .setTokenString(ZhConverterUtil.toSimple(origText));
        out.add(st);
    }
    return out;
}
```

`services.xml`:

```xml
<component id="ai.vespa.examples.linguistics.asia.OpenCcLinguistics"
           bundle="linguistics-asia-opencc"/>
```

## Switching to a region-specific direction

To target Hong Kong-specific Traditional → Simplified instead of generic, swap the static `ZhConverterUtil.toSimple` call for an explicit converter:

```java
// At class-level
private static final ZhConvert HK_TO_S = ZhConvertBootstrap.newInstance().convert("hk2s");

// In fold(...)
return HK_TO_S.convert(input);
```

(API surface depends on opencc4j version; see [opencc4j README](https://github.com/houbb/opencc4j#api) for the exact accessors. The shape above is the supported path for opencc4j 1.8.x.)

Or maintain a per-language map — call `t2s` for unknown Traditional and `hk2s` for `language=zh-HK` documents, etc. Plug `Token.getOrig()` and the per-token `LinguisticsParameters.language()` together to route.

## Build, test, deploy to Vespa Cloud

Sign up at <https://cloud.vespa.ai/>, then:

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-07     # replace TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# Cross-variant queries — both should hit cn-001 AND tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

## Inspecting the tokens

Use the `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) to confirm the OpenCC fold landed in the index:

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

Expect every `title_tokens` / `body_tokens` array to be Simplified only, regardless of source script.

## Local deploy + test

The local Docker flow doubles as the CI smoke test. The Vespa sample-apps CI parses the `data-test` attributes below: `&lt;pre data-test="exec"&gt;` blocks run as shell, and `data-test-assert-contains="STR"` requires `STR` in the captured stdout. The Full conventions in [parent README — Testing method](../README.md#testing-method).

Run from this sub-example directory:

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/07-opennlp-opencc
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

<pre data-test="exec" data-test-assert-contains="手机壳">
vespa query 'yql=select * from product where id contains "tw-001"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## Trade-offs vs ICU

| | OpenCC4j (07) | ICU (03 / 06) |
|---|---|---|
| Generic trad↔simp accuracy | high | high |
| HK / TW regional directions | yes (`s2hk`, `s2twp`, …) | no (`Traditional-Simplified` is one bucket) |
| Maintainer | community (Github) | Unicode Consortium |
| Used in published Vespa sample apps | no | yes (`lucene-linguistics` examples) |
| Bundle size impact | small (~512 KB) | medium (~13 MB for ICU dictionaries) |
| Languages beyond Chinese | Chinese-only | all of Unicode (Japanese normalize, Korean jamo, etc.) |

## References

- [opencc4j](https://github.com/houbb/opencc4j)
- [Original OpenCC C++ project (direction reference)](https://github.com/BYVoid/OpenCC)
- [06 — same wrapping pattern with ICU instead of OpenCC4j](../06-opennlp-icu)
- [03 — config-only ICU path](../03-lucene-icu)
