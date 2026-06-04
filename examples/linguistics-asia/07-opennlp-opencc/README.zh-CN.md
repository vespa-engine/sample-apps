<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

# 07 — 自定义 Java Linguistics + OpenCC4j（地区变体感知）

结构与 [06-opennlp-icu](../06-opennlp-icu) 相同，繁简折叠由 [OpenCC4j](https://github.com/houbb/opencc4j) 提供而非 ICU4J。在「通用繁↔简」用例上两者功能等价（本数据集）。选用 07 而非 06 的理由是 **更细粒度的中文地区变体**——ICU 单一的 `Traditional-Simplified` 桶不暴露这层。

## 选 07 还是 06 / 03

| 需求 | 选 |
|---|---|
| 通用繁↔简，配置即可 | [03-lucene-icu](../03-lucene-icu) |
| 通用繁↔简，要 Java 后门 | [06-opennlp-icu](../06-opennlp-icu) |
| **港式正体（zh-HK）与台式区分** | **07（本例）—— 或写一个包 OpenCC4j 的 Lucene factory（见说明）** |
| **地区专属短语对照**（台湾用语 vs 大陆用语） | **07** —— 或包 OpenCC4j 的 Lucene factory |
| **双向地区感知转换**（s2hk、hk2s、tw2s、s2twp …） | **07** —— 或包 OpenCC4j 的 Lucene factory |

> **Lucene 端替代方案。** OpenCC4j 是一个 library，不是接入点。07 把它包在 `OpenNlpLinguistics` 子类中；同样可以把它包成 Lucene `TokenFilterFactory`——就是 08 的 SPI factory 模式 + 07 的 `ZhConverterUtil` 呼叫的组合。同一个 library、Lucene 配置路径。我们没把这做成独立子例，因为它就是 07 与 08 的机械合并，但若你其它部分已选定 Lucene 路径并想留在其中，这是正确选择。

OpenCC 提供数个 ICU 没有的方向：

| OpenCC 方向 | 含义 | 何时需要 |
|---|---|---|
| `s2t` | 简体 → 繁体（通用） | 通用 |
| `s2tw` | 简体 → 台式繁体 | 台湾正字法 |
| `s2hk` | 简体 → 港式繁体 | 港式正字法 |
| `s2twp` | 简体 → 台式繁体 + 短语替换 | 地区用语替换 |
| `t2s` | 繁体 → 简体 | 本例所用 |
| `tw2s` | 台式繁体 → 简体 | `s2tw` 的反向 |
| `hk2s` | 港式繁体 → 简体 | `s2hk` 的反向 |

若你的商品同时包含 zh-TW *与* zh-HK 内容（同一商品两种正字法），且希望各自正确规范化，OpenCC 更合适。

若你只在意「繁→简，不计较哪种繁」，**留在 03 或 06** —— ICU 是 Vespa 内建，不必多管依赖，维护更简单。

## 两个类

[`OpenCcLinguistics.java`](src/main/java/ai/vespa/examples/linguistics/asia/OpenCcLinguistics.java) —— 结构与 06 的 `IcuLinguistics` 相同。继承 `OpenNlpLinguistics`，包装 tokenizer，覆盖 `getTokenizer` / `getStemmer` / `getSegmenter`。

[`OpenCcTokenizer.java`](src/main/java/ai/vespa/examples/linguistics/asia/OpenCcTokenizer.java) —— 逐 token 调用 `ZhConverterUtil.toSimple(originText)` 折叠。`getOrig()` ≠ `getTokenString()` 的模式与 06 相同（背后的 Vespa 行为见 [06 README](../06-opennlp-icu/README.zh-CN.md#关键模式在-token-上设-tokenstring不预先归一化整段输入)）。

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

`services.xml`：

```xml
<component id="ai.vespa.examples.linguistics.asia.OpenCcLinguistics"
           bundle="linguistics-asia-opencc"/>
```

## 切到地区专属方向

若要从「通用繁→简」改成「港式繁→简」，把 `ZhConverterUtil.toSimple` 换成显式 converter：

```java
// 类级
private static final ZhConvert HK_TO_S = ZhConvertBootstrap.newInstance().convert("hk2s");

// 在 fold(...) 内
return HK_TO_S.convert(input);
```

（API 形态依 opencc4j 版本而定；具体调用见 [opencc4j README](https://github.com/houbb/opencc4j#api)。上面这个形状适用 opencc4j 1.8.x。）

或维护一张按语言路由的对照表 —— 未知繁体走 `t2s`，`language=zh-HK` 文档走 `hk2s` 等。把 `Token.getOrig()` 与 `LinguisticsParameters.language()` 配着用即可。

## 编译、测试、部署到 Vespa Cloud

到 <https://cloud.vespa-cloud.com/> 注册免费试用，然后：

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-07     # 替换 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# 跨变体查询 —— 都应该同时命中 cn-001 与 tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

## 查看分词结果

用 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) 验证 OpenCC 折叠是否进了索引：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

预期：每个 `title_tokens` / `body_tokens` 数组无论源字形如何，都只含简体。

## 在本机部署 + 测试

本机 Docker 流程同时是 CI 烟雾测试。Vespa sample-apps CI 解析下方的 `data-test` 属性：`<pre data-test="exec">` block 当 shell 执行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整规范见 [parent README — Testing method](../README.zh-CN.md#testing-method)。

在本子例目录下执行：

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

## 与 ICU 的取舍

| | OpenCC4j（07） | ICU（03 / 06） |
|---|---|---|
| 通用繁简准确度 | 高 | 高 |
| HK / TW 地区方向 | 有（`s2hk`、`s2twp` 等） | 无（`Traditional-Simplified` 是一个桶） |
| 维护方 | 社区（Github） | Unicode Consortium |
| Vespa 公开 sample app 用过？ | 否 | 是（`lucene-linguistics` 例子） |
| Bundle 大小影响 | 小（~512 KB） | 中等（ICU 字典约 13 MB） |
| 中文以外的语言 | 仅中文 | 所有 Unicode（日文规范化、韩文谚文分解等） |

## 参考资料

- [opencc4j](https://github.com/houbb/opencc4j)
- [OpenCC C++ 原版（方向参考）](https://github.com/BYVoid/OpenCC)
- [06 —— 用 ICU 取代 OpenCC4j 的同一包装模式](../06-opennlp-icu)
- [03 —— 配置 only 的 ICU 路径](../03-lucene-icu)
