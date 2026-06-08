<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

# 08 — Lucene 詞組字典（自訂 SPI factory）

自訂的 Lucene `TokenFilterFactory` 加上配對的 `TokenFilter`，透過 `META-INF/services/` 註冊，讓 `services.xml` 可以像內建 filter 那樣使用 `phraseDict` 這個名稱。啟動時從 `configDir` 讀取詞組字典，運行時**把連續的 token 重新合回單一詞組 token**——只要這串 token 在字典裡。

這解決 ICU CJK 管線的一個具體問題：`icuTokenizer` 把 `空气净化器` 切成多個 token（類似 `[空气, 净化, 器]`）。下游依賴整詞辨識的 filter（同義詞查詢、關鍵詞保護、多 token 精確匹配）就看不到這個詞組是一個整體。字典裡加一條 `空气净化器`，本 filter 就在下一個 filter 跑之前把它合回單一 token。

## 何時選 08

| 需求 | 選 |
|---|---|
| 通用 ICU 繁↔簡，設定即可 | [03-lucene-icu](../03-lucene-icu) |
| ICU + 真實中文字典（CKIP / HanLP / IK / Jieba user-dict / 品牌詞表 / 商品分類詞） | **08（本例）** |
| 同樣效果但走 Java | [06-opennlp-icu](../06-opennlp-icu) |
| 地區感知（s2hk、s2twp…） | [07-opennlp-opencc](../07-opennlp-opencc) |

08 是「把任何文字型詞組字典經 SPI 接進 Lucene 分析鏈」的標準範本。把 `dictionary.txt` 換成中研院 CKIP、HanLP、IK 或內部品牌/品類字典輸出即可——形狀相同。

## 兩個類別

[`PhraseDictTokenFilter.java`](src/main/java/ai/vespa/examples/linguistics/asia/lucene/PhraseDictTokenFilter.java) —— 繼承 `TokenFilter`，標記 `final`（Lucene 契約）。緩衝輸入流後做貪婪最長匹配詞組掃描，最長匹配長度由字典中最長條目決定。

[`PhraseDictTokenFilterFactory.java`](src/main/java/ai/vespa/examples/linguistics/asia/lucene/PhraseDictTokenFilterFactory.java) —— 繼承 `TokenFilterFactory` 並實作 `ResourceLoaderAware`，透過 Vespa 的 resource loader 從 `configDir` 讀取字典檔。

SPI 註冊在 [`META-INF/services/org.apache.lucene.analysis.TokenFilterFactory`](src/main/resources/META-INF/services/org.apache.lucene.analysis.TokenFilterFactory)：

```
ai.vespa.examples.linguistics.asia.lucene.PhraseDictTokenFilterFactory
```

類的靜態 `NAME` 是 `phraseDict`——這就是 `services.xml` 中使用的名稱。

## services.xml 接線

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
    <!-- zh-TW: 同一條鏈 -->
  </analysis>
</config>
```

過濾器順序很重要。聚合器跑在 `icuTransform` **之後**，所以字典條目可以全用簡體撰寫——它們匹配的是摺疊之後的 token 流，無論源文件是 zh-CN 或 zh-TW。

## 字典檔

[`linguistics/dictionary.txt`](src/main/application/linguistics/dictionary.txt) —— 一行一個詞組，`#` 開頭是註解。本範例：

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

換成真實中文字典只需換掉這個檔。常見來源：

- **中研院 CKIP（中央研究院 CKIP 中文斷詞系統）** —— 把字典匯出為一行一詞的文字檔。
- **HanLP** —— `data/dictionary/*.txt` lexicon 檔。
- **IK Analyzer** —— `main2012.dic` 格式。
- **Jieba user-dict** —— 拿掉第三欄（詞頻/詞性）即可。
- **內部商品詞表** —— 商品名、品牌變體、型號等。

filter 對格式無要求：任何一行一詞的純文字檔（摺疊後形式）都能用。

## 建置、測試、部署到 Vespa Cloud

到 <https://cloud.vespa.ai/> 申請免費試用，然後：

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-08     # 替換 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# 多字詞組查詢能精確命中（單 token 匹配）。
vespa query 'yql=select * from product where default contains "空气净化器"' 'model.locale=zh-CN'
# -> cn-009（詞組在索引與查詢時都是單一 token）

vespa query 'yql=select * from product where default contains "空氣清淨機"' 'model.locale=zh-TW'
# -> tw-009（ICU 摺疊成 空气清净机 後聚合器合成單 token）

# 透過 ICU 的跨變體仍然有效
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
# -> cn-001 與 tw-001
```

## 檢視分詞結果

用 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) 確認聚合器把多字詞組收成單一 token：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

預期 `title_tokens` 陣列裡會看到 `[..., "空气净化器", ...]`、`[..., "笔记本电脑", ...]` —— 每個字典詞組都是一個元素，不會被切成單字 bigram 或詞碎片。

## 在本機部署 + 測試

本機 Docker 流程同時作為 CI 煙霧測試。Vespa sample-apps CI 解析下方的 `data-test` 屬性：`<pre data-test="exec">` block 當 shell 執行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整規範見 [parent README — Testing method](../README.zh-TW.md#testing-method)。

從本子例目錄執行：

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

## 搭配同義詞

要橋接地區詞彙差（`空气净化器` ↔ `空气清净机`、`笔记本电脑` ↔ `笔记型电脑`），把本聚合器與 [03-lucene-icu](../03-lucene-icu) 展示的同義詞 filter 模式配在一起用。聚合器給同義詞 filter 提供了乾淨的單 token 輸入。完整鏈：

```xml
<tokenFilters>
  <item><name>icuTransform</name>...</item>
  <item><name>phraseDict</name>...</item>      <!-- 08 -->
  <item><name>synonymGraph</name>...</item>     <!-- 03 -->
  <item><name>flattenGraph</name></item>
  <item><name>lowercase</name></item>
</tokenFilters>
```

實務上的行為取決於 Lucene 同義詞 filter 與圖替代路徑的互動——調參指南請見 03。

## 參考資料

- [Lucene `TokenFilterFactory` SPI](https://lucene.apache.org/core/9_8_0/analysis/common/org/apache/lucene/analysis/TokenFilterFactory.html)
- [Vespa Lucene linguistics](https://docs.vespa.ai/en/lucene-linguistics.html)
- 同層：[`examples/lucene-linguistics/add-token-filter-factory`](../../lucene-linguistics/add-token-filter-factory) —— 通用 Lucene SPI factory 範本（非中文專屬）
- [中研院 CKIP](https://ckip.iis.sinica.edu.tw/) | [HanLP](https://github.com/hankcs/HanLP) | [IK Analyzer](https://github.com/medcl/elasticsearch-analysis-ik) | [Jieba user-dict 格式](https://github.com/fxsjy/jieba#%E8%BD%BD%E5%85%A5%E8%AF%8D%E5%85%B8)
