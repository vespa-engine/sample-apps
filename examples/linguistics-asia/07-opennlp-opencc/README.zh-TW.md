<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

# 07 — 自訂 Java Linguistics + OpenCC4j（地區變體感知）

結構與 [06-opennlp-icu](../06-opennlp-icu) 相同，繁簡摺疊改由 [OpenCC4j](https://github.com/houbb/opencc4j) 提供而非 ICU4J。在「通用繁↔簡」用例兩者功能等價（本資料集）。選 07 而非 06 的理由是 **更細粒度的中文地區變體**——ICU 單一的 `Traditional-Simplified` 桶不暴露這層。

## 選 07 還是 06 / 03

| 需求 | 選 |
|---|---|
| 通用繁↔簡，設定即可 | [03-lucene-icu](../03-lucene-icu) |
| 通用繁↔簡，要 Java 後門 | [06-opennlp-icu](../06-opennlp-icu) |
| **港式繁體（zh-HK）與台式區分** | **07（本例）—— 或寫一個包 OpenCC4j 的 Lucene factory（見說明）** |
| **地區專屬詞彙對照**（台灣用語 vs 大陸用語） | **07** —— 或包 OpenCC4j 的 Lucene factory |
| **雙向地區感知轉換**（s2hk、hk2s、tw2s、s2twp …） | **07** —— 或包 OpenCC4j 的 Lucene factory |

> **Lucene 端替代方案。** OpenCC4j 是一個 library，不是接入點。07 把它包在 `OpenNlpLinguistics` 子類中；同樣可以把它包成 Lucene `TokenFilterFactory`——就是 08 的 SPI factory 模式 + 07 的 `ZhConverterUtil` 呼叫的組合。同一個 library、Lucene 設定路徑。我們沒把這做成獨立子例,因為它就是 07 與 08 的機械合併,但若你其餘部分已選定 Lucene 路徑並想留在其中,這是正確選擇。

OpenCC 提供數個 ICU 沒有的方向：

| OpenCC 方向 | 意思 | 何時需要 |
|---|---|---|
| `s2t` | 簡體 → 繁體（通用） | 通用 |
| `s2tw` | 簡體 → 台式繁體 | 台灣正字法 |
| `s2hk` | 簡體 → 港式繁體 | 港式正字法 |
| `s2twp` | 簡體 → 台式繁體 + 詞彙替換 | 地區用語替換 |
| `t2s` | 繁體 → 簡體 | 本例使用 |
| `tw2s` | 台式繁體 → 簡體 | `s2tw` 的反向 |
| `hk2s` | 港式繁體 → 簡體 | `s2hk` 的反向 |

若你的商品同時含 zh-TW *與* zh-HK 內容（同一商品兩種正字法），且希望各自正確規範化，OpenCC 較合適。

若你只在意「繁→簡，不在乎是哪種繁」，**留在 03 或 06** —— ICU 是 Vespa 內建，不必多管依賴，維護較簡單。

## 兩個類別

[`OpenCcLinguistics.java`](src/main/java/ai/vespa/examples/linguistics/asia/OpenCcLinguistics.java) —— 結構與 06 的 `IcuLinguistics` 相同。繼承 `OpenNlpLinguistics`，包裝 tokenizer，覆寫 `getTokenizer` / `getStemmer` / `getSegmenter`。

[`OpenCcTokenizer.java`](src/main/java/ai/vespa/examples/linguistics/asia/OpenCcTokenizer.java) —— 逐 token 呼叫 `ZhConverterUtil.toSimple(originText)` 摺疊。`getOrig()` ≠ `getTokenString()` 模式與 06 相同（背後 Vespa 行為見 [06 README](../06-opennlp-icu/README.zh-TW.md#關鍵模式在-token-上設-tokenstring不預先正規化整段輸入)）。

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

## 切到地區專屬方向

若要從「通用繁→簡」改成「港式繁→簡」，把 `ZhConverterUtil.toSimple` 換成顯式 converter：

```java
// 類級
private static final ZhConvert HK_TO_S = ZhConvertBootstrap.newInstance().convert("hk2s");

// 在 fold(...) 內
return HK_TO_S.convert(input);
```

（API 形狀依 opencc4j 版本而定；具體呼叫見 [opencc4j README](https://github.com/houbb/opencc4j#api)。上面形狀適用 opencc4j 1.8.x。）

或維護一張依語言路由的對照表 —— 未知繁體走 `t2s`，`language=zh-HK` 文件走 `hk2s` 等。把 `Token.getOrig()` 與 `LinguisticsParameters.language()` 搭配起來路由即可。

## 建置、測試、部署到 Vespa Cloud

到 <https://cloud.vespa.ai/> 申請免費試用，然後：

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-07     # 替換 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# 跨變體查詢 —— 都應該同時命中 cn-001 與 tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

## 檢視分詞結果

用 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) 確認 OpenCC 摺疊有寫入索引：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

預期：每個 `title_tokens` / `body_tokens` 陣列無論來源字形為何，都只含簡體。

## 在本機部署 + 測試

本機 Docker 流程同時作為 CI 煙霧測試。Vespa sample-apps CI 解析下方的 `data-test` 屬性：`<pre data-test="exec">` block 當 shell 執行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整規範見 [parent README — Testing method](../README.zh-TW.md#testing-method)。

從本子例目錄執行：

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

## 與 ICU 的取捨

| | OpenCC4j（07） | ICU（03 / 06） |
|---|---|---|
| 通用繁簡準確度 | 高 | 高 |
| HK / TW 地區方向 | 有（`s2hk`、`s2twp` 等） | 無（`Traditional-Simplified` 是一個桶） |
| 維護方 | 社群（Github） | Unicode Consortium |
| Vespa 公開 sample app 用過？ | 否 | 是（`lucene-linguistics` 範例） |
| Bundle 大小影響 | 小（~512 KB） | 中等（ICU 字典約 13 MB） |
| 中文以外語言 | 僅中文 | 所有 Unicode（日文正規化、韓文諺文分解等） |

## 參考資料

- [opencc4j](https://github.com/houbb/opencc4j)
- [OpenCC C++ 原專案（方向參考）](https://github.com/BYVoid/OpenCC)
- [06 —— 同樣包裝模式但用 ICU 取代 OpenCC4j](../06-opennlp-icu)
- [03 —— config-only ICU 路徑](../03-lucene-icu)
