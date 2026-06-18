<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

# 04 — Reuse the existing Jieba bundle, no new Java

Vespa is happy to load any `com.yahoo.language.Linguistics` implementation that ships in a container-plugin bundle. The sibling sample app [vespa-chinese-linguistics](../../vespa-chinese-linguistics) already provides one — a Jieba-based segmenter wrapping OpenNLP. This sub-example is the *consumer* side: a pure application package (no `pom.xml`) that picks up the prebuilt bundle and turns it on with three lines of services.xml.

Pick this option when you want **Jieba-quality Simplified Chinese word segmentation** with dictionary support, and you don't want to write or maintain your own Linguistics module.

## How it plugs in

`services.xml` instantiates the class from the existing bundle:

```xml
<component id="com.qihoo.language.JiebaLinguistics"
           bundle="vespa-chinese-linguistics">
  <config name="com.qihoo.language.config.jieba">
    <!-- optional dictionary + stopwords paths, relative to app root -->
  </config>
</component>
```

That's the entire mechanism for plugging in a custom Linguistics: `<component class="..." bundle="..."/>`. The `bundle` attribute must match the JAR's bundle-symbolic-name (`vespa-chinese-linguistics`), and the JAR must live under `app/components/` when you deploy.

## Build the sibling bundle once

```sh
# From the sample-apps root
( cd ../vespa-chinese-linguistics && mvn package )
cp ../vespa-chinese-linguistics/target/vespa-chinese-linguistics-1.0.0-deploy.jar \
   app/components/
```

You don't need to rebuild it for every deploy of this app — only when the Jieba bundle source changes.

## Deploy to Vespa Cloud

Sign up at <https://cloud.vespa.ai/>, then:

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-04     # replace TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

# Jieba segments "无线蓝牙耳机" into ["无线", "蓝牙", "耳机"]
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' \
            'model.locale=zh-CN' 'trace.level=2'
```

Use `trace.level=2` to confirm Jieba's tokenization at query time. For index-side inspection, query the `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens):

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

The `title_tokens` array shows Jieba's segmentation, e.g. `["无线", "蓝牙", "耳机"]`.

## Local deploy + test

The local Docker flow doubles as the CI smoke test. The Vespa sample-apps CI parses the `data-test` attributes below: `&lt;pre data-test="exec"&gt;` blocks run as shell, and `data-test-assert-contains="STR"` requires `STR` in the captured stdout. The Full conventions in [parent README — Testing method](../README.md#testing-method).

Run from this sub-example directory:

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/04-jieba
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 vespaengine/vespa
vespa config set target local
vespa status deploy --wait 300
# Build the sibling Jieba bundle once and copy into app/components/.
( cd ../../vespa-chinese-linguistics && mvn -q package )
cp ../../vespa-chinese-linguistics/target/vespa-chinese-linguistics-1.0.0-deploy.jar app/components/
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

<pre data-test="exec" data-test-assert-contains="蓝牙">
vespa query 'yql=select * from product where id contains "cn-002"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## What you get and what you don't

- **Simplified Chinese segmentation**: Jieba's dictionary handles common compounds well — better than OpenNLP CJK bigrams, comparable to Lucene SmartCN.
- **Custom dictionary support**: drop a `.dict` file in the application package and reference it from the `<config>` block. Useful for product names and brand terms.
- **Traditional Chinese**: Jieba's built-in dictionary is Simplified-only. zh-TW input gets segmented but with worse results than zh-CN. If you need Traditional, combine this with [06-opennlp-icu](../06-opennlp-icu)'s ICU fold, or pick [03-lucene-icu](../03-lucene-icu).
- **No code to maintain in this app** — the only thing in this directory is the application package.

## References

- Sibling: [vespa-chinese-linguistics](../../vespa-chinese-linguistics)
- [Linguistics interface](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java)
- [Jieba (jieba-analysis)](https://github.com/huaban/jieba-analysis)
