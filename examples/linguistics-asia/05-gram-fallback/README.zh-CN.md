<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

# 05 — Gram 兜底方案，不用任何 linguistics 组件

有时候正确答案就是：别在分词上做文章。Vespa schema 提供 `match: gram` 模式，在索引和查询时都把字段切成固定长度的重叠字符 n-gram。不需要 `Linguistics` 组件、不需要词典、不需要分词模型。任何 CJK 输入都能跑，任何 Unicode 输入都能跑，因为它只看字符。

适用场景：
- 想要**最大召回**，可以接受一点过度匹配。
- 不能或不愿意发布 Java bundle。
- 索引里混了多种变体或脚本（zh-CN、zh-TW、日文汉字等），一套规则无法覆盖。

## 这个 app 做了什么

`services.xml` 没有声明任何 `Linguistics` 组件。Vespa 仍然加载内建 OpenNLP，但 gram 匹配的字段会绕过它，完全由 schema 的 match 规则切。

`app/schemas/product.sd` 的关键部分：

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

“无线蓝牙耳机”被切成 bigram：`无线`、`线蓝`、`蓝牙`、`牙耳`、`耳机`。任何含某个 bigram 的文档都能被相应查询召回。BM25 排序照常工作——倒排索引不在乎一个 token 是“词”还是“2-gram”。

## 部署到 Vespa Cloud

到 <https://cloud.vespa-cloud.com/> 注册免费试用，然后：

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-05     # 替换 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

预期：每个查询命中本变体的文档，并可能因 bigram 重叠串到另一变体。这是它的取舍。

## 查看分词结果

schema 中宣告了 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，查询它即可看到实际写入索引的 2-gram：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

`title_tokens` 数组会显示重叠 2-gram（简体如 `手机`、`机壳`；繁体如 `手機`、`機殼`）。

## 在本机部署 + 测试

本机 Docker 流程同时是 CI 烟雾测试。Vespa sample-apps CI 解析下方的 `data-test` 属性：`<pre data-test="exec">` block 当 shell 执行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整规范见 [parent README — Testing method](../README.zh-CN.md#testing-method)。

在本子例目录下执行：

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

## 取舍

- **索引会变大**：每个字符大约一条 posting。小目录没问题（本例笔记本上跑无压力）；大目录要做容量规划。
- **没有词干化、停用词、同义词**：bigram 上的 BM25 是粗活，但对小型 zh-CN 商品目录有时强过把品牌名切坏的“聪明”分词器。
- **变体串流**：简繁共享的 bigram 会跨匹配；但码点不同的 bigram（如 `机` vs `機`）依然分开。所以 05 只算半个繁简互通方案。
- **拉丁字母不受影响**：gram 规则只切汉字，连续的拉丁字母仍作整体 token。

要更紧的召回语义，参见 [01-opennlp-cjk](../01-opennlp-cjk)（同样思路，但其余非 CJK 文本走完整 OpenNLP 管线），或升级到 [02-lucene-per-variant](../02-lucene-per-variant)（词级分词）。

## 参考资料

- [Vespa text matching — gram](https://docs.vespa.ai/en/querying/text-matching.html#gram)
- [Schema reference — `match`](https://docs.vespa.ai/en/reference/schema-reference.html#match)
