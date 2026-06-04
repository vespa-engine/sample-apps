<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

# 03 — Lucene linguistics + ICU 繁簡互通

[02-lucene-per-variant](../02-lucene-per-variant) 的「跨變體」表親。一樣的 `LuceneLinguistics` 接線,分析鏈改為:

1. **`icuTokenizer`** —— Unicode 感知的 CJK 詞級斷詞(取代 SmartCN;字典隨 Lucene `lucene-analysis-icu` JAR 一起出貨)。
2. **`icuTransform` 帶 `id=Traditional-Simplified`** —— ICU 標準的繁→簡轉寫器。索引與查詢兩側同樣應用,於是 zh-CN 查詢能命中 zh-TW 文件、反之亦然。
3. `lowercase` 給一起進來的拉丁 token。

無自訂 Java，Lucene 之外無第三方依賴。

## 02 vs 03 選擇

| | [02-lucene-per-variant](../02-lucene-per-variant) | **03 —— 本例** |
|---|---|---|
| zh-CN ↔ zh-TW 召回 | 隔離(獨立店面) | 統一(一個店面服務兩個市場) |
| Tokenizer | zh-CN 用 SmartCN,zh-TW 用 Standard+cjkBigram | 兩者都用 ICU CJK 詞分詞 |
| 跨變體摺疊 | 無 | 有(ICU transliterator) |
| Java 程式碼 | 無 | 無 |
| 建置 | Maven(smartcn 依賴) | Maven(icu 依賴) |

## 分析鏈接入

`services.xml`:

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

`zh-TW` 項目完全相同——這就是本例的核心,轉寫過後變體標籤已無意義。

`<conf><item key="id">Traditional-Simplified</item></conf>` 是 lucene-linguistics 給 SPI 工廠傳參的語法,與 [lucene-linguistics samples](../../lucene-linguistics/going-crazy) 中其他例子用的 `<conf>` 形狀一致。

## 建置並部署到 Vespa Cloud

到 <https://cloud.vespa-cloud.com/> 申請免費試用，然後：

```sh
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-03     # 替換 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# 跨變體查詢 —— 都應該同時命中 cn-001 與 tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'

# CJK 詞級切分 —— 應同時命中 cn-002(无线蓝牙耳机)與 tw-002(無線藍牙耳機)
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' 'model.locale=zh-CN'
```

## 檢視分詞結果

用 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) 確認繁→簡摺疊真的寫入索引：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

每篇文件的 `title_tokens` / `body_tokens` 應只含簡體字元——zh-TW 文件的原繁體文字仍會出現在 `title` / `body` 欄位，但不會出現在 `_tokens` 陣列中。

## 在本機部署 + 測試

本機 Docker 流程同時作為 CI 煙霧測試。Vespa sample-apps CI 解析下方的 `data-test` 屬性：`<pre data-test="exec">` block 當 shell 執行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整規範見 [parent README — Testing method](../README.zh-TW.md#testing-method)。

從本子例目錄執行：

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

## 用同義詞補詞彙缺口

轉寫處理字形差異(`手機殼` ↔ `手机壳`),**不**處理地區詞彙差異——zh-CN 的 `空气净化器` 與 zh-TW 的 `空氣清淨機` **都摺成簡體後**仍是 `空气净化器` vs `空气清净机`,詞不同。鏈上用 Lucene `synonymGraph` 過濾器(配 `flattenGraph` 讓候選寫進索引):

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

[`linguistics/synonyms.txt`](src/main/application/linguistics/synonyms.txt) —— Solr 同義詞格式,逗號分隔的同義組。詞表**全用簡體**即可,因為轉寫步驟已把所有內容折回簡體。

`configDir`(在 `<config>` 頂端宣告)告訴 `LuceneLinguistics` 資源檔相對應用套件根目錄的路徑。

### 已驗證

```sh
vespa query 'yql=select * from product where default contains "锅"' 'model.locale=zh-CN'
# → 同時命中 电饭煲(cn-008)與 电子鍋(tw-008)—— 單字同義 锅 ⇔ 煲 跨越詞彙差。
```

### 多字 CJK 同義詞注意事項

`tokenizerFactory=…ICUTokenizerFactory` 讓同義詞檔按執行期一樣的 ICU CJK 規則切分,理論上 `空气净化器,空气清净机` 這類多字組也該命中。實務上 ICU 的詞界啟發可能讓你的同義詞項切出的 token 序列與文件/查詢略有差異,同義詞就永遠找不到匹配。若多字組沒按預期橋接:

1. `vespa query ... summary=debug-tokens` —— 看執行期文字實際怎麼切的（回應中的 `title_tokens` / `body_tokens` 陣列）。
2. `trace.level=2` —— 看查詢端怎麼切的。
3. 調整同義詞檔:用顯式空格強制 token 邊界(如 `空 气 净 化 器, 空 气 清 净 机`),或對關鍵場景退回到單字對。

正式部署通常把這個檔當作可調整的產物,依真實查詢 log 迭代調校。

## 注意事項

- ICU 的 `Traditional-Simplified` 轉寫是單向(把繁體摺到簡體)。`summary` 欄位仍呈現文件原字形——只有索引被規範化。
- 港式繁體(zh-HK)與台式繁體不完全相同;ICU 的 `Traditional-Simplified` 涵蓋大多情境。若需港式特例,ICU 還有 `s2hk`、`t2hk` 可串接。
- ICU tokenizer 用基於字典的斷詞。簡體下品質接近 SmartCN,繁體下顯著優於 `cjkBigram`。

## 參考資料

- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- [Lucene `lucene-analysis-icu`](https://lucene.apache.org/core/9_8_0/analysis/icu/index.html)
- [ICU Transliterator IDs](https://unicode-org.github.io/icu/userguide/transforms/general/#transliterator-identifiers)
