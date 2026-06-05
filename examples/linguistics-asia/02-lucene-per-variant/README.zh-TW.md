<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

# 02 — Lucene linguistics，每個中文變體一套分析鏈

`LuceneLinguistics` 允許依語言代碼各自組合 Apache Lucene 的分析鏈。本子例採用**逐變體獨立**的接線方式：`zh-CN` 文件與查詢走 SmartCN（基於 HMM 的簡體斷詞器）；`zh-TW` 文件與查詢走 Standard 分詞器 + CJK bigram 過濾器。兩條鏈互不干擾。

適用情境：希望針對某個變體拿到**正經的詞級分詞**，並且**zh-CN 與 zh-TW 是兩個獨立市場**，不希望搜尋結果互相串連。

需要繁簡互通召回（zh-TW 查詢命中 zh-CN 文件）？請看 [03-lucene-icu](../03-lucene-icu)，在相同的 Lucene 鏈前加上一個 OpenCC `CharFilter`，斷詞前先把繁體摺回簡體。

## 這個 app 做了什麼

`services.xml` 在 `LuceneLinguistics` 的 analysis 對應表中列出三個項目：

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

- **zh-CN** → SmartCN 的 `hmmChinese` 輸出詞級 token：`无线蓝牙耳机` → `无线`、`蓝牙`、`耳机`。這裡不做停用詞過濾——Lucene 沒有 `smartChineseStop` SPI；如需停用詞，請用通用 `stop` filter 搭配自己的停用詞清單。
- **zh-TW** → SmartCN 的訓練語料是簡體，對繁體切分品質很差。所以這條鏈用 `standard` 按碼點切，再用 `cjkBigram` 展開成重疊的 2-gram。Solr/Elasticsearch 處理繁體常用此法。
- **不要再加 `zh` 兜底項** —— `zh` 與 `zh-CN` 都解析為 `Language.CHINESE_SIMPLIFIED`，兩條 `<item>` 同時存在會讓後者覆蓋前者。只用明確的變體鍵即可。

## 建置並部署到 Vespa Cloud

到 <https://cloud.vespa-cloud.com/> 申請免費試用，然後：

```sh
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-02     # 替換 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# zh-CN 查詢，由 SmartCN 進行詞切
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' \
            'model.locale=zh-CN' 'trace.level=2' \
  | jq '.trace.children | last | .children[] | select(.message) | select(.message | test("YQL.*")) | .message'

# zh-TW 查詢，走 bigram
vespa query 'yql=select * from product where default contains "無線藍牙耳機"' \
            'model.locale=zh-TW'
```

`trace.level=2` 可看到每條分析鏈對查詢的改寫：SmartCN 輸出三個詞；zh-TW 鏈輸出一串重疊 bigram。

## 檢視分詞結果

schema 宣告 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，查詢即可拿到每個欄位的 token 流，不需登入容器：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

zh-CN 文件的 `title_tokens` 應是詞級切分（`无线`、`蓝牙`、`耳机`）；zh-TW 文件的 `title_tokens` 應是 bigram（`無線`、`線藍`、`藍牙` ...）。

## 在本機部署 + 測試

本機 Docker 流程同時作為 CI 煙霧測試。Vespa sample-apps CI 解析下方的 `data-test` 屬性：`<pre data-test="exec">` block 當 shell 執行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整規範見 [parent README — Testing method](../README.zh-TW.md#testing-method)。

從本子例目錄執行：

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

## 為什麼不直接用 SmartCN 處理繁體

SmartCN 的 HMM 是在簡體語料上訓練的。把繁體丟給它，切出來的邊界看似合理但與文件中實際用字不一致。CJK bigram 簡單粗糙但穩定。若想在繁體上拿到 SmartCN 同等品質，務實作法是先摺回簡體——請參見 [03-lucene-icu](../03-lucene-icu)。

## 參考資料

- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- [Lucene `smartcn` analyzer](https://lucene.apache.org/core/9_8_0/analysis/smartcn/index.html)
- [Lucene `cjkBigram` filter](https://lucene.apache.org/core/9_8_0/analysis/common/org/apache/lucene/analysis/cjk/CJKBigramFilter.html)
