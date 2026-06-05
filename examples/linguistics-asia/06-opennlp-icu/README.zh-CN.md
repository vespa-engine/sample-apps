<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

# 06 — 自定义 Java Linguistics（模板 + ICU 折叠）

约 70 行 Java 写一个自定义 `Linguistics`：继承 Vespa 的 `OpenNlpLinguistics`，包一层 `Tokenizer`，对每个 token 调 ICU `Traditional-Simplified` 把原文折成简体并写到该 token 的 `tokenString`。zh-CN 与 zh-TW 在索引和查询两侧都能互通召回。

这就是「在 Vespa 中需要自定义 Linguistics 时该照搬的模板」。结构（继承 `OpenNlpLinguistics`、覆盖 `getTokenizer`/`getStemmer`/`getSegmenter`、包装父类的 tokenizer）与你想做的具体变换无关。这里的 ICU 繁简折叠只是示例载荷。

如果只是要繁简互通、又不想发布自定义 Java bundle，[03-lucene-icu](../03-lucene-icu) 用纯 XML（Lucene `icuTransform`）做到同样效果。选 06 是当你具体想要 **OpenNLP-extends 形态** —— 在一个类里同时暴露 Vespa 完整 Linguistics API（`Tokenizer` / `Stemmer` / `Segmenter` / `Detector`），而不是 Lucene 分析链。大多数任意 Java 逻辑——品牌词保护、自定义词典查询、按 token 呼叫模型服务等——同样可以塞进 Lucene `TokenFilterFactory` 内（形状见 [08](../08-lucene-phrase-dict)）；06 与 08 之间的选择主要是你想扩 Vespa 的哪个面。

## 两个类

[`IcuLinguistics.java`](src/main/java/ai/vespa/examples/linguistics/asia/IcuLinguistics.java)：

```java
public class IcuLinguistics extends OpenNlpLinguistics {
    private final Tokenizer tokenizer;

    public IcuLinguistics() {
        super();
        this.tokenizer = new IcuTokenizer(super.getTokenizer());
    }

    @Override public Tokenizer getTokenizer()  { return tokenizer; }
    @Override public Stemmer   getStemmer()    { return new StemmerImpl(tokenizer); }
    @Override public Segmenter getSegmenter()  { return new SegmenterImpl(tokenizer); }
}
```

三个覆盖，都返回**同一个** wrapper 实例。`getTokenizer` 用于索引时管线；`getStemmer` / `getSegmenter` 用于查询侧。三者要一致——若其中一个返回父类的原始 tokenizer，那条路径就绕过了你的包装。

### 索引侧 vs 查询侧需要不同行为

若希望两侧切词不同（例如索引时切得宽以提升召回，查询时切得紧；或反之），建两个 `Tokenizer` 实例分别注入：

```java
public IcuLinguistics() {
    this.indexTokenizer = new IcuTokenizer(super.getTokenizer(), Mode.INDEX);
    this.queryTokenizer = new IcuTokenizer(super.getTokenizer(), Mode.SEARCH);
}

@Override public Tokenizer getTokenizer() { return indexTokenizer; }
@Override public Stemmer   getStemmer()   { return new StemmerImpl(queryTokenizer); }
@Override public Segmenter getSegmenter() { return new SegmenterImpl(queryTokenizer); }
```

这是 [`JiebaLinguistics`](../../vespa-chinese-linguistics/src/main/java/com/qihoo/language/JiebaLinguistics.java) 用的模式：Jieba 的 `SegMode.INDEX` 切得较宽、重叠较多，`SegMode.SEARCH` 切得较紧，专为查询解析使用。

OpenCC / ICU 折叠两侧做同样的事，所以我们用一个 tokenizer 实例。两侧确实需要差异时再用双模式形状。

[`IcuTokenizer.java`](src/main/java/ai/vespa/examples/linguistics/asia/IcuTokenizer.java)：

```java
public class IcuTokenizer implements Tokenizer {
    private static final Transliterator T2S = Transliterator.getInstance("Traditional-Simplified");
    private final Tokenizer delegate;

    public IcuTokenizer(Tokenizer delegate) { this.delegate = delegate; }

    private Iterable<Token> wrap(Iterable<Token> raw) {
        List<Token> out = new ArrayList<>();
        for (Token t : raw) {
            String origText = t.getOrig();
            SimpleToken st = new SimpleToken(origText)
                    .setOffset(t.getOffset())
                    .setType(t.getType())
                    .setTokenString(T2S.transliterate(origText));
            out.add(st);
        }
        return out;
    }

    @Override public Iterable<Token> tokenize(String input, LinguisticsParameters parameters) {
        return wrap(delegate.tokenize(input, parameters));
    }

    @Override @SuppressWarnings("deprecation")
    public Iterable<Token> tokenize(String input, Language language, StemMode stemMode, boolean removeAccents) {
        return wrap(delegate.tokenize(input, language, stemMode, removeAccents));
    }
}
```

`services.xml`：

```xml
<component id="ai.vespa.examples.linguistics.asia.IcuLinguistics"
           bundle="linguistics-asia-icu"/>
```

## 关键模式：在 token 上设 `tokenString`，**不**预先归一化整段输入

在**原始**输入上做切词，然后逐 token 设 `tokenString` 为变换后的值。**不要**把整段输入先做归一化再交给 delegate。每个 token 必须满足：

| | 设为 | 为什么 |
|---|---|---|
| `getOrig()` | 原始子串（繁体 `蘋果`） | Vespa 用它作为该 token offset+length 的源文本参照 |
| `getTokenString()` | 变换后形式（简体 `苹果`） | 索引与匹配的「搜索键」 |

两者必须**不同**，变换在索引时才会生效。当它们相等时，Vespa 把索引形式存为 `null`（一个「不覆写」的优化，含义是「用原始字段子串」），这会悄悄抵消你在输入层做的归一化。上面的模式保证两者不同。

[`JiebaTokenizer`](../../vespa-chinese-linguistics/src/main/java/com/qihoo/language/JiebaTokenizer.java) 用的就是这个形状：`new SimpleToken(originToken).setTokenString(segmentedWord)`。

## 编译、测试、部署到 Vespa Cloud

到 <https://cloud.vespa-cloud.com/> 注册免费试用，然后：

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-06     # 替换 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# 跨变体查询 —— 都应该同时命中 cn-001 与 tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

## 验证索引中只剩简体

查询 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，读响应里的 `title_tokens` / `body_tokens`：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

预期：每个 `title_tokens` 数组只含简体字符（如 `手机壳`、`苹果`），即使文档的 `title` summary 字段仍是繁体。ICU 折叠发生在索引时的 token 流上——源文本 summary 不变。

## 在本机部署 + 测试

本机 Docker 流程同时是 CI 烟雾测试。Vespa sample-apps CI 解析下方的 `data-test` 属性：`<pre data-test="exec">` block 当 shell 执行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整规范见 [parent README — Testing method](../README.zh-CN.md#testing-method)。

在本子例目录下执行：

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/06-opennlp-icu
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

## 与其它子例对比

| 子例 | 跨变体 | 自定义 Java | Maven |
|---|---|---|---|
| [02-lucene-per-variant](../02-lucene-per-variant) | 否（隔离） | 否 | 是 |
| [03-lucene-icu](../03-lucene-icu) | **是**（Lucene `icuTransform`） | 否 | 是 |
| [04-jieba](../04-jieba) | 否 | 否（复用同级 bundle） | 否 |
| **06 —— 本例** | **是**（Java 端 ICU4J） | **是** | 是 |

在本数据集上，**03 与 06 的跨变体效果等价**。能用配置就用 03；要 Java 侧的自由就用 06。

## 参考资料

- [Vespa `Linguistics` 接口](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java)
- [同级示例 vespa-chinese-linguistics（相同包装模式，Jieba 载荷）](../../vespa-chinese-linguistics)
- [ICU4J Transliterator IDs](https://unicode-org.github.io/icu/userguide/transforms/general/#transliterator-identifiers)
