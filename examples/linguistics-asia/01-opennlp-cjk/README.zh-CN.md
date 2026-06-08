<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

# 01 — OpenNLP 启用 CJK 分词

Vespa 容器自带 `OpenNlpLinguistics`，默认不处理中日韩文。打开两个 config 开关后，中文会按相邻 2 字符切成 bigram。无需写 Java、无需 Maven，仅 XML 配置。

适用场景：zh-CN 商品目录较小、对分词质量要求不高时，最便宜的起步方案。

## 这个 app 做了什么

- `services.xml` 声明 `OpenNlpLinguistics`，并设置 `<cjk>true</cjk>` 与 `<createCjkGrams>true</createCjkGrams>`。
- 中文文本被切成相邻 2 字的 gram：`无线蓝牙耳机` → `无线`、`线蓝`、`蓝牙`、`牙耳`、`耳机`。查询词做同样处理，BM25 能正常工作。
- 拉丁文按 OpenNLP 默认方式切词。

## 局限

- **不区分字形脚本**：简体和繁体走同一套 bigram，只看 Unicode 码点。zh-CN 查询 `手机壳` 不会命中 zh-TW 文档中的 `手機殼`。
- **无词干化、无停用词、无同义词**：只做切位置。
- **索引会变大**：每段 N 字文本会产生约 N 条 posting。

若上述任何一点是问题，请看 [02-lucene-per-variant](../02-lucene-per-variant)（词级分词）、[04-jieba](../04-jieba)（Jieba 词典分词），或 [06-opennlp-icu](../06-opennlp-icu)（繁简互通召回）。

## 部署到 Vespa Cloud

到 <https://cloud.vespa.ai/> 注册免费试用，然后：

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-01     # 替换 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

# 查询 —— 简体词只命中简体文档
vespa query 'yql=select * from product where default contains "手机壳"' \
            'model.locale=zh-CN'

# 同款的繁体文档 —— 这里查不到
vespa query 'yql=select * from product where default contains "手機殼"' \
            'model.locale=zh-TW'
```

完整对比查询见 [../ext/queries.md](../ext/queries.md)。

## 查看分词结果

schema 中宣告了 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，逐字段暴露 token 流。查询它、读 `title_tokens` / `body_tokens`：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

简体文档会看到 `手机`、`机壳` 之类 2-gram；繁体则是 `手機`、`機殼` —— 码点不同、bigram 不同，不互相命中。

## 在本机部署 + 测试

本机 Docker 流程同时是 CI 烟雾测试。Vespa sample-apps CI 解析下方的 `data-test` 属性：`<pre data-test="exec">` block 当 shell 执行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整规范见 [parent README — Testing method](../README.zh-CN.md#testing-method)。

在本子例目录下执行：

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

## 参考资料

- [Vespa OpenNLP linguistics](https://docs.vespa.ai/en/linguistics/linguistics-opennlp.html)
- [Vespa 简报 — OpenNLP CJK 支持](https://blog.vespa.ai/vespa-newsletter-august-2024/)
