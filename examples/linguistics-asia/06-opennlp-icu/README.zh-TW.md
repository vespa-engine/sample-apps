<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

# 06 — 自訂 Java Linguistics（範本 + ICU 摺疊）

約 70 行 Java 寫一個自訂 `Linguistics`：繼承 Vespa 的 `OpenNlpLinguistics`，包一層 `Tokenizer`，對每個 token 用 ICU `Traditional-Simplified` 把原文摺成簡體並寫入該 token 的 `tokenString`。zh-CN 與 zh-TW 在索引和查詢兩側都能互通召回。

這就是「在 Vespa 中需要自訂 Linguistics 時該照抄的範本」。結構（繼承 `OpenNlpLinguistics`、覆寫 `getTokenizer`/`getStemmer`/`getSegmenter`、包裝父類的 tokenizer）與你想做的具體變換無關。這裡的 ICU 繁簡摺疊只是示例載荷。

如果只是要繁簡互通、又不想發布自訂 Java bundle，[03-lucene-icu](../03-lucene-icu) 用純 XML（Lucene `icuTransform`）做到同樣效果。挑 06 是當你具體想要 **OpenNLP-extends 形態** —— 在一個類別中同時暴露 Vespa 完整 Linguistics API（`Tokenizer` / `Stemmer` / `Segmenter` / `Detector`），而非 Lucene 分析鏈。大多任意 Java 邏輯——品牌詞保護、自訂字典查詢、按 token 呼叫模型服務等——同樣可塞進 Lucene `TokenFilterFactory` 內（形狀見 [08](../08-lucene-phrase-dict)）；06 與 08 之間的選擇主要看你想擴 Vespa 的哪個面。

## 兩個類別

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

三個覆寫，全部回傳**同一個** wrapper 實例。`getTokenizer` 用於索引時管線；`getStemmer` / `getSegmenter` 用於查詢側。三者要保持一致——若其中一個回傳父類的原始 tokenizer，那條路徑就繞過了你的包裝。

### 索引側 vs 查詢側需要不同行為

若希望兩側斷詞不同（例如索引時切得寬以提升召回，查詢時切得緊；或反之），建兩個 `Tokenizer` 實例分別注入：

```java
public IcuLinguistics() {
    this.indexTokenizer = new IcuTokenizer(super.getTokenizer(), Mode.INDEX);
    this.queryTokenizer = new IcuTokenizer(super.getTokenizer(), Mode.SEARCH);
}

@Override public Tokenizer getTokenizer() { return indexTokenizer; }
@Override public Stemmer   getStemmer()   { return new StemmerImpl(queryTokenizer); }
@Override public Segmenter getSegmenter() { return new SegmenterImpl(queryTokenizer); }
```

這是 [`JiebaLinguistics`](../../vespa-chinese-linguistics/src/main/java/com/qihoo/language/JiebaLinguistics.java) 用的模式：Jieba 的 `SegMode.INDEX` 切得較寬、重疊較多，`SegMode.SEARCH` 切得較緊，專為查詢解析使用。

OpenCC / ICU 摺疊兩側做同樣的事，所以我們用一個 tokenizer 實例。兩側確實需要差異時再用雙模式形狀。

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

## 關鍵模式：在 token 上設 `tokenString`，**不**預先正規化整段輸入

在**原始**輸入上做斷詞，然後逐 token 設 `tokenString` 為變換後的值。**不要**把整段輸入先做正規化再交給 delegate。每個 token 必須滿足：

| | 設為 | 為什麼 |
|---|---|---|
| `getOrig()` | 原始子字串（繁體 `蘋果`） | Vespa 用它作為該 token offset+length 的源文本對照 |
| `getTokenString()` | 變換後形式（簡體 `苹果`） | 索引與匹配的「搜尋鍵」 |

兩者必須**不同**，變換才會在索引時生效。當它們相等時，Vespa 把索引形式存為 `null`（「不覆寫」的最佳化，意即「用原始欄位子字串」），這會悄悄抵消你在輸入層的正規化。上面的模式保證兩者不同。

[`JiebaTokenizer`](../../vespa-chinese-linguistics/src/main/java/com/qihoo/language/JiebaTokenizer.java) 用的就是這個形狀：`new SimpleToken(originToken).setTokenString(segmentedWord)`。

## 建置、測試、部署到 Vespa Cloud

到 <https://cloud.vespa-cloud.com/> 申請免費試用，然後：

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-06     # 替換 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# 跨變體查詢 —— 都應該同時命中 cn-001 與 tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

## 驗證索引中只剩簡體

查詢 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，讀回應裡的 `title_tokens` / `body_tokens`：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

預期：每個 `title_tokens` 陣列只含簡體字元（如 `手机壳`、`苹果`），即使該文件的 `title` summary 欄位仍是繁體。ICU 摺疊發生在索引時的 token 流上——原文 summary 不變。

## 在本機部署 + 測試

本機 Docker 流程同時作為 CI 煙霧測試。Vespa sample-apps CI 解析下方的 `data-test` 屬性：`<pre data-test="exec">` block 當 shell 執行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整規範見 [parent README — Testing method](../README.zh-TW.md#testing-method)。

從本子例目錄執行：

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

## 與其他子例對比

| 子例 | 跨變體 | 自訂 Java | Maven |
|---|---|---|---|
| [02-lucene-per-variant](../02-lucene-per-variant) | 否（隔離） | 否 | 是 |
| [03-lucene-icu](../03-lucene-icu) | **是**（Lucene `icuTransform`） | 否 | 是 |
| [04-jieba](../04-jieba) | 否 | 否（重用同層 bundle） | 否 |
| **06 —— 本例** | **是**（Java 端 ICU4J） | **是** | 是 |

在本資料集上，**03 與 06 的跨變體效果等價**。能用設定就用 03；需要 Java 端自由度就用 06。

## 參考資料

- [Vespa `Linguistics` 介面](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java)
- [同層範例 vespa-chinese-linguistics（相同包裝模式，Jieba 載荷）](../../vespa-chinese-linguistics)
- [ICU4J Transliterator IDs](https://unicode-org.github.io/icu/userguide/transforms/general/#transliterator-identifiers)
