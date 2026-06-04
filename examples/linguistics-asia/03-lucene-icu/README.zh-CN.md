<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

# 03 — Lucene linguistics + ICU 繁简互通

[02-lucene-per-variant](../02-lucene-per-variant) 的「跨变体」表亲。同样的 `LuceneLinguistics` 接线，分析链改为：

1. **`icuTokenizer`** —— Unicode 感知的 CJK 词级分词（替代 SmartCN；词典随 Lucene `lucene-analysis-icu` JAR 一起出货）。
2. **`icuTransform` 带 `id=Traditional-Simplified`** —— ICU 的标准繁→简转写器。索引与查询两侧同样应用，所以 zh-CN 查询命中 zh-TW 文档，反之亦然。
3. `lowercase` 给一起进来的拉丁 token。

无自定义 Java，无 Lucene 之外的第三方库。

## 02 vs 03 选择

| | [02-lucene-per-variant](../02-lucene-per-variant) | **03 —— 本例** |
|---|---|---|
| zh-CN ↔ zh-TW 召回 | 隔离（独立店面） | 统一（一店服务两市场） |
| Tokenizer | zh-CN 用 SmartCN，zh-TW 用 Standard+cjkBigram | 两者都用 ICU CJK 词分词 |
| 跨变体折叠 | 无 | 有（ICU transliterator） |
| Java 代码 | 无 | 无 |
| 构建 | Maven（smartcn 依赖） | Maven（icu 依赖） |

## 分析链接入

`services.xml`：

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

`zh-TW` 条目与上完全相同——这正是本例的意义所在，转写过后变体标签不再有意义。

`<conf><item key="id">Traditional-Simplified</item></conf>` 是 lucene-linguistics 给 SPI 工厂传参的语法，与 [lucene-linguistics samples](../../lucene-linguistics/going-crazy) 里其它例子用的 `<conf>` 形状一致。

## 编译并部署到 Vespa Cloud

到 <https://cloud.vespa-cloud.com/> 注册免费试用，然后：

```sh
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-03     # 替换 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# 跨变体查询 —— 都应该同时命中 cn-001 与 tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'

# CJK 词级切分 —— 应同时命中 cn-002（无线蓝牙耳机）与 tw-002（無線藍牙耳機）
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' 'model.locale=zh-CN'
```

## 查看分词结果

用 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) 验证繁→简折叠真的进了索引：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

每篇文档的 `title_tokens` / `body_tokens` 应只含简体字符——zh-TW 文档的原繁体文字仍会出现在 `title` / `body` 字段，但不会出现在 `_tokens` 数组里。

## 在本机部署 + 测试

本机 Docker 流程同时是 CI 烟雾测试。Vespa sample-apps CI 解析下方的 `data-test` 属性：`<pre data-test="exec">` block 当 shell 执行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整规范见 [parent README — Testing method](../README.zh-CN.md#testing-method)。

在本子例目录下执行：

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

## 用同义词补词汇缺口

转写处理字形差异（`手機殼` ↔ `手机壳`），**不**处理地区词汇差异——zh-CN 的 `空气净化器` 与 zh-TW 的 `空氣清淨機` **都折成简体后**仍是 `空气净化器` vs `空气清净机`，词不同。链上用 Lucene `synonymGraph` 过滤器（加 `flattenGraph` 让候选写入索引）：

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

[`linguistics/synonyms.txt`](src/main/application/linguistics/synonyms.txt) —— Solr 同义词格式，逗号分隔同义组。词表**全用简体**写就好，转写步骤已把所有内容折回简体。

`configDir`（在 `<config>` 顶部声明）告诉 `LuceneLinguistics` 资源文件相对应用包根目录的路径。

### 已验证

```sh
vespa query 'yql=select * from product where default contains "锅"' 'model.locale=zh-CN'
# → 同时命中 电饭煲（cn-008）与 电子鍋（tw-008）—— 单字同义词 锅 ⇔ 煲 跨过词汇差。
```

### 多字 CJK 同义词注意事项

`tokenizerFactory=…ICUTokenizerFactory` 让同义词文件按运行时一样的 ICU CJK 规则切分，理论上 `空气净化器,空气清净机` 这类多字组也该命中。实务上 ICU 的词界启发可能让你的同义词项切出的 token 序列与文档/查询略有差异，同义词就永远找不到匹配。若多字组没按预期桥接：

1. `vespa query ... summary=debug-tokens` —— 看运行时文本实际怎么切的（响应中的 `title_tokens` / `body_tokens` 数组）。
2. `trace.level=2` —— 看查询侧怎么切的。
3. 调整同义词文件：用显式空格强制 token 边界（如 `空 气 净 化 器, 空 气 清 净 机`），或者对关键场景退回到单字对。

生产部署通常把这个文件当作可调整的产物，依真实查询日志迭代调优。

## 注意事项

- ICU 的 `Traditional-Simplified` 转写是单向（把繁体折到简体）。`summary` 字段仍渲染文档原字形——只有索引被规范化。
- 港式正体（zh-HK）与台式繁体不完全一致；ICU 的 `Traditional-Simplified` 已覆盖大部分情形。如需港式特例，ICU 还有 `s2hk`、`t2hk` 可串接。
- ICU tokenizer 用基于词典的分词。简体下质量接近 SmartCN，繁体下显著优于 `cjkBigram`。

## 参考资料

- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- [Lucene `lucene-analysis-icu`](https://lucene.apache.org/core/9_8_0/analysis/icu/index.html)
- [ICU Transliterator IDs](https://unicode-org.github.io/icu/userguide/transforms/general/#transliterator-identifiers)
