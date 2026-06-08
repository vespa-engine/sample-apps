<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

# 08 — Lucene 短语字典（自定义 SPI factory）

一个自定义的 Lucene `TokenFilterFactory` 加配对的 `TokenFilter`，通过 `META-INF/services/` 注册，让 `services.xml` 可以像内建 filter 那样用 `phraseDict` 这个名字。启动时从 `configDir` 读取短语字典，运行时**把连续的 token 重新合回单一短语 token**——只要这串 token 在字典里出现过。

这解决 ICU CJK 管线的一个具体问题：`icuTokenizer` 把 `空气净化器` 切成多个 token（类似 `[空气, 净化, 器]`）。下游依赖整词的 filter（同义词查询、关键词保护、多 token 精确匹配）就看不到这个短语是一个整体。字典里加一条 `空气净化器`，本 filter 就在下一个 filter 跑之前把它合回单 token。

## 何时选 08

| 需求 | 选 |
|---|---|
| 通用 ICU 繁↔简，配置即可 | [03-lucene-icu](../03-lucene-icu) |
| ICU + 真实中文字典（CKIP / HanLP / IK / Jieba user-dict / 品牌词表 / 商品类目词） | **08（本例）** |
| 同样效果但走 Java | [06-opennlp-icu](../06-opennlp-icu) |
| 地区感知（s2hk、s2twp…） | [07-opennlp-opencc](../07-opennlp-opencc) |

08 是「把任意基于文本的短语字典经 SPI 接进 Lucene 分析链」的标准模板。把 `dictionary.txt` 换成中研院 CKIP、HanLP、IK 或内部品牌/类目字典输出即可——形状不变。

## 两个类

[`PhraseDictTokenFilter.java`](src/main/java/ai/vespa/examples/linguistics/asia/lucene/PhraseDictTokenFilter.java) —— 继承 `TokenFilter`，标记 `final`（Lucene 契约）。缓冲输入流后做贪心最长匹配短语扫描，最长匹配长度由字典里最长条目决定。

[`PhraseDictTokenFilterFactory.java`](src/main/java/ai/vespa/examples/linguistics/asia/lucene/PhraseDictTokenFilterFactory.java) —— 继承 `TokenFilterFactory` 并实现 `ResourceLoaderAware`，通过 Vespa 的 resource loader 从 `configDir` 读取字典文件。

SPI 注册在 [`META-INF/services/org.apache.lucene.analysis.TokenFilterFactory`](src/main/resources/META-INF/services/org.apache.lucene.analysis.TokenFilterFactory)：

```
ai.vespa.examples.linguistics.asia.lucene.PhraseDictTokenFilterFactory
```

类的静态 `NAME` 是 `phraseDict`——这就是 `services.xml` 中使用的名字。

## services.xml 接线

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
    <!-- zh-TW: 同一条链 -->
  </analysis>
</config>
```

过滤器顺序很重要。聚合器跑在 `icuTransform` **之后**，所以字典条目可以全部用简体写——它们匹配的是折叠之后的 token 流，无论源文档是 zh-CN 还是 zh-TW。

## 字典文件

[`linguistics/dictionary.txt`](src/main/application/linguistics/dictionary.txt) —— 一行一个短语，`#` 开头是注释。本示例：

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

要换成真实的中文字典只需替换这个文件。常见来源：

- **中研院 CKIP（中央研究院 CKIP 中文斷詞系統）** —— 把字典导出成一行一词的文本。
- **HanLP** —— `data/dictionary/*.txt` lexicon 文件。
- **IK Analyzer** —— `main2012.dic` 格式。
- **Jieba user-dict** —— 去掉第三栏（频率/词性）即可。
- **内部商品词表** —— 商品名、品牌变体、型号等。

filter 对格式无要求：任何一行一词的纯文本（折叠后形式）都能用。

## 编译、测试、部署到 Vespa Cloud

到 <https://cloud.vespa.ai/> 注册免费试用，然后：

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-08     # 替换 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# 多字短语查询能精确命中（单 token 匹配）。
vespa query 'yql=select * from product where default contains "空气净化器"' 'model.locale=zh-CN'
# -> cn-009（短语在索引与查询时都是单个 token）

vespa query 'yql=select * from product where default contains "空氣清淨機"' 'model.locale=zh-TW'
# -> tw-009（ICU 折叠成 空气清净机 后聚合器合成单 token）

# 通过 ICU 的跨变体仍然有效
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
# -> cn-001 与 tw-001
```

## 查看分词结果

用 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) 确认聚合器把多字短语收成了单个 token：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

预期 `title_tokens` 数组里看到 `[..., "空气净化器", ...]`、`[..., "笔记本电脑", ...]` —— 每个词典短语都是一个元素，不被切成单字 bigram 或词碎片。

## 在本机部署 + 测试

本机 Docker 流程同时是 CI 烟雾测试。Vespa sample-apps CI 解析下方的 `data-test` 属性：`<pre data-test="exec">` block 当 shell 执行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整规范见 [parent README — Testing method](../README.zh-CN.md#testing-method)。

在本子例目录下执行：

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

## 搭配同义词

要桥接地区词汇差（`空气净化器` ↔ `空气清净机`、`笔记本电脑` ↔ `笔记型电脑`），把本聚合器与 [03-lucene-icu](../03-lucene-icu) 里展示的同义词 filter 模式配在一起用。聚合器给同义词 filter 提供了干净的单 token 输入。完整链型：

```xml
<tokenFilters>
  <item><name>icuTransform</name>...</item>
  <item><name>phraseDict</name>...</item>      <!-- 08 -->
  <item><name>synonymGraph</name>...</item>     <!-- 03 -->
  <item><name>flattenGraph</name></item>
  <item><name>lowercase</name></item>
</tokenFilters>
```

实务上的行为取决于 Lucene 同义词 filter 与图替代路径的交互——调参指南见 03。

## 参考资料

- [Lucene `TokenFilterFactory` SPI](https://lucene.apache.org/core/9_8_0/analysis/common/org/apache/lucene/analysis/TokenFilterFactory.html)
- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- 同层：[`examples/lucene-linguistics/add-token-filter-factory`](../../lucene-linguistics/add-token-filter-factory) —— 通用 Lucene SPI factory 模板（非中文专属）
- [中研院 CKIP](https://ckip.iis.sinica.edu.tw/) | [HanLP](https://github.com/hankcs/HanLP) | [IK Analyzer](https://github.com/medcl/elasticsearch-analysis-ik) | [Jieba user-dict 格式](https://github.com/fxsjy/jieba#%E8%BD%BD%E5%85%A5%E8%AF%8D%E5%85%B8)
