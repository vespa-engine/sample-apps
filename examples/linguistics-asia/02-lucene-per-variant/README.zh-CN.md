<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

# 02 — Lucene linguistics，每个中文变体一套分析链

`LuceneLinguistics` 允许按语言代码独立组合 Apache Lucene 的分析链。本子例采用**每变体独立**的接线方式：`zh-CN` 文档与查询走 SmartCN（基于 HMM 的简体词分词器）；`zh-TW` 文档与查询走 Standard 分词器 + CJK bigram 过滤器。两条链互不干扰。

适合场景：希望对其中一种变体获得**正经的词级分词**，并且**zh-CN 与 zh-TW 是两个独立市场**，不想让结果互相串。

需要繁简互通召回（zh-CN 查询命中 zh-TW 文档）？请看 [03-lucene-icu](../03-lucene-icu)，在同样的 Lucene 链前加一段 OpenCC `CharFilter`，分词前先把繁体折成简体。

## 这个 app 做了什么

`services.xml` 在 `LuceneLinguistics` 的 analysis 映射里给出三条配置：

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

- **zh-CN** → SmartCN 的 `hmmChinese` 输出词级 token：`无线蓝牙耳机` → `无线`、`蓝牙`、`耳机`。这里不做停用词过滤——Lucene 没有 `smartChineseStop` SPI；要停用词请用通用的 `stop` filter 配合自己的停用词表。
- **zh-TW** → SmartCN 训练语料是简体，对繁体切分质量很差。所以这一支用 `standard` 按码点切，再用 `cjkBigram` 展成重叠的 2-gram。这是 Solr/Elasticsearch 处理繁体的常见做法。
- **不要再加 `zh` 兜底项** —— `zh` 与 `zh-CN` 都解析为 `Language.CHINESE_SIMPLIFIED`，两条 `<item>` 同时出现会让后者覆盖前者。只用具体的变体键即可。

## 编译并部署到 Vespa Cloud

到 <https://cloud.vespa.ai/> 注册免费试用，然后：

```sh
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-02     # 替换 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# zh-CN 查询，由 SmartCN 做词切
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' \
            'model.locale=zh-CN' 'trace.level=2' \
  | jq '.trace.children | last | .children[] | select(.message) | select(.message | test("YQL.*")) | .message'

# zh-TW 查询，走 bigram
vespa query 'yql=select * from product where default contains "無線藍牙耳機"' \
            'model.locale=zh-TW'
```

`trace.level=2` 可以看到每条分析链对查询的重写：SmartCN 输出三个词；zh-TW 链输出一串重叠 bigram。

## 查看分词结果

schema 中宣告了 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，查询它即可读到每个字段的 token 流，不需要进容器：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

zh-CN 文档的 `title_tokens` 应是词级切分（`无线`、`蓝牙`、`耳机`）；zh-TW 文档的 `title_tokens` 应是 bigram（`無線`、`線藍`、`藍牙` ...）。

## 在本机部署 + 测试

本机 Docker 流程同时是 CI 烟雾测试。Vespa sample-apps CI 解析下方的 `data-test` 属性：`<pre data-test="exec">` block 当 shell 执行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整规范见 [parent README — Testing method](../README.zh-CN.md#testing-method)。

在本子例目录下执行：

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

## 为什么不在繁体上直接用 SmartCN

SmartCN 的 HMM 是在简体语料上训练的。把繁体喂给它，切出来的边界看似合理但与实际文档中的词不一致。CJK bigram 简单粗糙但稳定。若想在繁体上拿到 SmartCN 同等质量，现实路径是先折回简体——见 [03-lucene-icu](../03-lucene-icu)。

## 参考资料

- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- [Lucene `smartcn` analyzer](https://lucene.apache.org/core/9_8_0/analysis/smartcn/index.html)
- [Lucene `cjkBigram` filter](https://lucene.apache.org/core/9_8_0/analysis/common/org/apache/lucene/analysis/cjk/CJKBigramFilter.html)
