<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Linguistics-Asia：Vespa 中文 linguistics 選型指南

Vespa 處理中文有幾種方式 —— 簡體、繁體，以及你在 zh-CN、zh-TW、zh-HK 市場常見的混合情況。本目錄把這些選項並排比較。提供**八個可執行的子應用**——每種接入形態一個——共用同一份資料集與同一組對比查詢，讓取捨可以被觀察，而不是只停留在文件上。

每個子例都附三語 README（English、简体中文、繁體中文）。共享資料集與查詢放在 [`ext/`](ext/)。

## 為什麼斷詞決定了中文檢索

中文沒有空白，所以 tokenizer 的切分策略**就是**你的匹配策略。查詢 `藍牙耳機` 可能變成一個 token（命中字典）、三個 token（`藍`、`牙`、`耳機`——錯誤切分）、或五個重疊 bigram（召回寬、精準低）——端看用哪個分析器。簡繁還多一個維度：同款商品、碼點不同，沒摺疊腳本就沒有跨變體召回。

## 怎麼選

針對同樣八個範例的兩種視角加一張決策表。挑你團隊思考問題時貼近的那一邊即可。

### 決策表

| 選項 | 變體 | 詞級分詞 | 跨變體 | 需要 Maven | 適用情境 |
|---|---|---|---|---|---|
| [01 OpenNLP CJK](01-opennlp-cjk) | zh-CN + zh-TW（bigram） | 否 | 否 | 否 | 最小可跑的 zh-CN 應用，零建置 |
| [02 Lucene 逐變體](02-lucene-per-variant) | zh-CN + zh-TW 隔離 | SmartCN 處理 zh-CN | 否 | 是 | 獨立的 zh-CN 與 zh-TW 店面 |
| [03 Lucene 跨變體](03-lucene-icu) | zh-CN + zh-TW 統一 | ICU CJK 詞分詞 | 是（Lucene `icuTransform`） | 是 | 一個店面服務兩個市場，無自訂 Java |
| [04 Jieba 重用](04-jieba) | zh-CN | Jieba 字典 | 否 | 否（重用同層 bundle） | 需要自訂字典的 zh-CN |
| [05 Gram 兜底](05-gram-fallback) | 任意 CJK | 否 | 部分（只限碼點重疊 bigram） | 否 | 多腳本混合、最大召回、零建置 |
| [06 自訂 Java Linguistics](06-opennlp-icu) | zh-CN + zh-TW 統一 | OpenNLP CJK + 逐 token ICU 摺疊 | 是（Java ICU4J） | 是 | 任意自訂 Java Linguistics 的範本 |
| [07 OpenCC 地區變體](07-opennlp-opencc) | zh-CN + zh-TW + zh-HK 統一 | OpenNLP CJK + 逐 token OpenCC4j 摺疊 | 是（Java OpenCC4j） | 是 | 需要地區感知方向（s2hk、s2twp、hk2s 等）時再選 |
| [08 Lucene 詞組字典](08-lucene-phrase-dict) | zh-CN + zh-TW 統一 | ICU CJK + 自訂詞組聚合 | 是（Lucene SPI factory） | 是 | 要把真實中文字典（CKIP、HanLP、IK、品牌詞表）接進 Lucene 時再選 |

### 視角 A —— 依機制（你想怎麼接）

四種機制家族，依成本遞增：

**1. 純設定（無 Java）。** 在 `services.xml` 中宣告 Vespa 內建的 Linguistics。最小用 OpenNLP；要豐富的分析鏈選 Lucene。
- 最小部署 → **[01](01-opennlp-cjk)**（OpenNLP，字元 bigram）。
- 詞級分詞、變體獨立 → **[02](02-lucene-per-variant)**（Lucene）。
- 詞級分詞、變體摺成一個索引 → **[03](03-lucene-icu)**（Lucene）。
- 召回優先、任何 CJK 輸入 → **[05](05-gram-fallback)**（無 Linguistics、schema 層 `match: gram`）。

**2. 重用既有 bundle。** 把別人寫好的 `container-plugin` JAR 丟進 `app/components/`。本應用零 Java。
- Jieba 字典斷詞 → **[04](04-jieba)**。

**3. 自訂 Lucene SPI factory。** 自己寫 `TokenFilterFactory` / `TokenizerFactory` / `CharFilterFactory`，透過 `META-INF/services/` 註冊。繼續走 Lucene 設定路徑；factory 內部可以跑任意 Java。
- 詞組字典（CKIP、HanLP、IK、品牌詞表、內部商品詞表）→ **[08](08-lucene-phrase-dict)**。
- 同樣形狀還適用：DB 來源同義詞、基於 ML 的 filter、包 OpenCC4j 拿到 `s2hk` / `s2twp` / `hk2s` 方向、按 token 呼叫規範化服務等。

**4. 自訂 OpenNLP-extends Linguistics。** 繼承 `OpenNlpLinguistics`，覆寫 `getTokenizer` / `getStemmer` / `getSegmenter`（可選 `getDetector`）。提供 Vespa 完整 Linguistics API。
- ICU4J transliterator 範例 → **[06](06-opennlp-icu)**。
- OpenCC4j 範例（地區感知方向）→ **[07](07-opennlp-opencc)**。

機制 3 與 4 能力大致重疊；選擇是結構性而非能力差距。**預設選 3（Lucene factory）**，除非你具體需要（a）完整 Linguistics API 含 `Detector` / `Segmenter`，（b）與既有 OpenNLP-extends bundle 緊密整合，或（c）團隊已經投資在 OpenNLP-extends 形狀上。詳細取捨見下方 [兩套底層機制](#兩套底層機制) 段。

### 視角 B —— 依情境（你的資料與市場長什麼樣）

**涵蓋的市場**
- 單一市場，只 zh-CN → **[01-opennlp-cjk](01-opennlp-cjk)**（最小）、**[02-lucene-per-variant](02-lucene-per-variant)**（更好的斷詞）、或 **[04-jieba](04-jieba)**（之後想擴 Jieba 字典）。
- 單一市場，只 zh-TW → **[02-lucene-per-variant](02-lucene-per-variant)** 的 zh-TW 鏈，或 **[05-gram-fallback](05-gram-fallback)**（可接受寬召回）。
- 兩市場、隔離（獨立 zh-CN / zh-TW 店面） → **[02-lucene-per-variant](02-lucene-per-variant)**。
- 兩市場、統一搜尋（一個店面服務 zh-CN + zh-TW） → **[03-lucene-icu](03-lucene-icu)**（設定）或 **[06-opennlp-icu](06-opennlp-icu)**（Java）。
- 三市場含 zh-HK 港式正字法 → **[07-opennlp-opencc](07-opennlp-opencc)**，或包 OpenCC4j 的 Lucene factory。
- 多腳本混合（中 + 日 + 韓 + 拉丁）召回優先 → **[05-gram-fallback](05-gram-fallback)**。

**資料特徵**
- 商品目錄大，含品牌名、型號、產業用詞，ICU 易過度切分 → **[08-lucene-phrase-dict](08-lucene-phrase-dict)**（接真實中文詞組字典）。
- 地區詞彙差（如 笔记本电脑 vs 笔记型电脑） → **03 / 06 / 07 / 08** 任一，搭配一層 synonym。
- 混合大小寫的拉丁品牌（`iPhone`、`MacBook Pro`）出現在中文文字中 → 所有範例都正常處理（拉丁 token 正常 lowercase）。

**團隊 / 建置限制**
- 無 Java 能力、CI 無 Maven → **[01-opennlp-cjk](01-opennlp-cjk)**、**[04-jieba](04-jieba)**、或 **[05-gram-fallback](05-gram-fallback)**（純 XML / 複製 JAR）。
- Java 團隊，偏 Lucene 生態 → **02 / 03 / 08**。
- Java 團隊，偏 OpenNLP-extends 形狀 → **06 / 07**。
- 最小可部署 PoC，之後再回頭擴 → **[01-opennlp-cjk](01-opennlp-cjk)**。

### 關於 zh-HK（粵語）

Vespa 沒有官方的 zh-HK 或 yue 斷詞器。三個務實路徑：

1. **把 zh-HK 視為 zh-TW** —— 讓 zh-HK 走 03/06 的同一條鏈（摺疊後都是簡體）。多數電商文字適用，因為字形差異是主導，港版繁體與台版相近。
2. **需要港式正字法獨立處理**，要用 OpenCC 的 `s2hk` / `hk2s` 方向。ICU 沒有 `Traditional-HongKong` 轉寫器,所以單用 `icuTransform`（03 路徑）無法區分 HK 與 TW 繁體。把 OpenCC 的 HK 方向接進來有兩條路:
   - **Java 擴充 Linguistics** —— [07-opennlp-opencc](07-opennlp-opencc) 把 OpenCC4j 包在 `OpenNlpLinguistics` 子類別中。
   - **Lucene SPI factory 包 OpenCC4j** —— 我們沒做成獨立子例,但它就是 08 的 `TokenFilterFactory` 模式 + 07 的 `ZhConverterUtil` 呼叫的組合。同一個 library、不同接入點。若你希望其餘部分繼續走 Lucene 設定路徑,選這條。
3. **退回 gram 匹配（05）** —— 按碼點切，與 locale 無關，召回為先。

## 兩套底層機制

知道你要哪種接入路徑之後（視角 A 已給出），本節是這些選擇背後的更深說明。

### OpenNLP vs Lucene —— 該用哪一套

Vespa 內建兩套可插拔的 Linguistics 實作。這兩者之間的選擇決定了其他一切。

| | **OpenNLP 系**（預設，`OpenNlpLinguistics`） | **Lucene 系**（`LuceneLinguistics`） |
|---|---|---|
| 開箱即用的 CJK | 只有 bigram（`cjk=true` 旗標） | 字典詞斷詞（ICU、SmartCN 等） |
| 可用的 token filter | 極少 | ~100 個（synonym、stopword、n-gram、phonetic、ICU 等） |
| 語言涵蓋 | 英文 + 基本西文，其他較弱 | 內建 ~40 種語言 analyzer |
| 多階段分析鏈 | 需要 Java | 宣告式 XML |
| 依 locale 路由 | 一個元件，語言無感 | `<analysis><item key="...">` 對映 |
| 多套 analysis profile | 否 | 是 —— 每個欄位或用途一條鏈（見 [Lucene linguistics 文件](https://docs.vespa.ai/en/lucene-linguistics.html)） |
| 自訂路徑 | 用 Java 繼承 `OpenNlpLinguistics`（Jieba 模式） | 寫 Lucene SPI factory（`TokenFilterFactory` 等） |
| 是否需要 Maven | 預設不需要；擴充需要 | 需要 |
| Bundle 大小影響 | 極小（已載入） | 中等（每個 `lucene-analysis-*` 多 1-15 MB） |
| 本專案的子例 | 01、04、06、07 | 02、03、08 |

**預設選 Lucene**，除非你有具體理由用 OpenNLP。理由：

1. **中文變體場景特別受惠於 Lucene 生態**。ICU 的 CJK 詞斷詞與 `Traditional-Simplified` 轉寫在 Lucene 是純設定；用 OpenNLP 要達到同樣效果就要寫 Java（見 06）。
2. **多語言部署用 Lucene 較好擴展**。要支援 zh + ja + ko + th，只是在 `services.xml` 多寫三條 `<item>`；OpenNLP 每個非預設語言都意味多寫一個 Java 擴充。
3. **同義詞、停用詞、字典查詢、自訂 factory —— 都自然納入 Lucene 鏈**。OpenNLP 擴充容易變成單一用途的整塊程式碼，因為 OpenNLP 沒暴露同等的鏈組合 API。
4. **Lucene 路徑文件齊全、產業內已驗證**，不只在 Vespa 內。

OpenNLP 才是正確選擇的情況：

- **內建英文 + bigram CJK 已經夠** —— 不必發布 Maven 插件。01 就是一個 XML。
- **重用已發布的 OpenNLP-extends bundle**，例如 [vespa-chinese-linguistics](../vespa-chinese-linguistics) —— 04 就是把 JAR 丟進 `components/`。
- **想要 Vespa 完整 Linguistics API 一次暴露** —— Detector、Stemmer、Segmenter、Tokenizer 都在一個類別內。OpenNLP-extends 模式提供四個覆寫點；Lucene factory 只暴露 analyzer 鏈。例:06、07。

第三點是**偏好**，不是能力差距。把第三方 library（OpenCC4j、ML 服務、自訂字典）包進 Lucene `TokenFilterFactory` 同樣可行 —— 見 08 的 SPI factory 形狀，以及 [關於 zh-HK 段](#關於-zh-hk粵語) 的具體說明。若你其餘部分已選定 Lucene 設定路徑，即使是非內建行為也優先用 Lucene。

### 透過 Lucene SPI factory 自訂

中文變體場景下，內建 Lucene SPI（`icuTokenizer`、`icuTransform`、`synonym` 等）已涵蓋大多需求 —— 見 [02](02-lucene-per-variant) 與 [03](03-lucene-icu)。若需要一個不存在的 filter 或 tokenizer，就寫一個 Lucene `TokenFilterFactory` / `TokenizerFactory` / `CharFilterFactory`，透過 `META-INF/services/` 註冊，然後在 `services.xml` 以名稱引用。

本目錄內 [08-lucene-phrase-dict](08-lucene-phrase-dict) 是實戰範例 —— 自訂的詞組聚合 filter，啟動時讀取字典檔（CKIP / HanLP / IK / Jieba user-dict / 內部品牌詞表），把過度切分的 CJK token 流重新合回單一詞組 token。

非中文相關的通用 Lucene SPI factory 模式（DB 來源同義詞、基於 ML 的 filter、phonetic 演算法等），同層的 [lucene-linguistics](../lucene-linguistics) 範例已有額外範本：[`add-token-filter-factory`](../lucene-linguistics/add-token-filter-factory) 與 [`custom-analyzer-phonetic`](../lucene-linguistics/custom-analyzer-phonetic)。

### 透過 OpenNLP-extends Linguistics 自訂

機制就是 `services.xml` 內一條 component 宣告：

```xml
<component id="my.package.MyLinguistics" bundle="my-bundle"/>
```

類別實作 [`com.yahoo.language.Linguistics`](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java)。實務上繼承 `OpenNlpLinguistics` 並覆寫 `getTokenizer()`、`getStemmer()`、`getSegmenter()`，三者回傳同一個包裝後的 tokenizer。[06-opennlp-icu](06-opennlp-icu) 是最小端到端範例；[vespa-chinese-linguistics](../vespa-chinese-linguistics) 是另一份可工作的 bundle（Jieba）。

## 八個子例

所有子例共用同一份 `product.sd` schema 形狀與同一份資料集（`ext/documents.jsonl`），結果可直接對比——跑 `ext/queries.md` 的對比查詢即可。

### Schema 要點：`set_language` 欄位順序

每個子例 schema 都有：

```
field language type string {
  indexing: summary | attribute | set_language
  match: word
}
```

`set_language` 告訴索引器對文件中**後續**欄位套用哪種語言的 Linguistics 規則。**`language` 欄位必須在任何需要按語言斷詞的欄位之前宣告**（這裡是 `title` 與 `body`）。indexing pipeline 依宣告順序處理欄位；若 `language` 放在 `title`/`body` 之後，`set_language` 觸發時已經太晚，索引器退回容器預設（English，OpenNLP CJK），不論欄位值是什麼。容易破、部署時沒警告、餵資料後才以「召回差」表現出來。

### 目錄

- **[01-opennlp-cjk](01-opennlp-cjk)** —— 內建 OpenNLP linguistics，啟用 CJK 設定旗標。無程式碼、無 Maven，僅 XML。基於 bigram。
- **[02-lucene-per-variant](02-lucene-per-variant)** —— `LuceneLinguistics`，每個 locale 一套分析鏈：zh-CN 用 SmartCN，zh-TW 用 Standard + CJK bigram。在重要處用詞級分詞；變體保持隔離。
- **[03-lucene-icu](03-lucene-icu)** —— Lucene `icuTokenizer`（CJK 詞分詞）+ 帶 `id=Traditional-Simplified` 的 `icuTransform` token filter。索引與查詢兩側都摺成簡體，於是 zh-CN ↔ zh-TW 互相召回。純 XML 設定，無自訂 Java。
- **[04-jieba](04-jieba)** —— 純應用套件，直接重用預先建好的 [`vespa-chinese-linguistics`](../vespa-chinese-linguistics) bundle。示範 `<component bundle="..."/>` 接入方式，本子例不含 Java。
- **[05-gram-fallback](05-gram-fallback)** —— 完全不宣告 Linguistics 元件；`match: gram, gram-size: 2` 在 schema 中完成所有事。任何 Unicode 輸入（包括多腳本混合）都能跑。
- **[06-opennlp-icu](06-opennlp-icu)** —— 自訂 Java `Linguistics`，繼承 `OpenNlpLinguistics`，包裝其 tokenizer，用 ICU4J 的 `Transliterator` 逐 token 做繁→簡摺疊。效果與 03 等價，在 Java 層做。OpenNLP-extends 形態的標準範本——當你想要 Vespa 完整 Linguistics API（Tokenizer / Stemmer / Segmenter / Detector）而非 Lucene 分析鏈時用。同樣繁簡效果但走 Lucene SPI 請看 03；任意 Java 邏輯塞進 Lucene 鏈請看 08。
- **[07-opennlp-opencc](07-opennlp-opencc)** —— 與 06 同樣的自訂 Java 形狀,但以 OpenCC4j 取代 ICU4J。僅在需要 OpenCC 的地區感知方向（`s2hk`、`s2twp`、`hk2s` 等）時再用 ——ICU 單一 `Traditional-Simplified` 不暴露這層。一般繁簡情境 ICU（03 或 06）成本較低。
- **[08-lucene-phrase-dict](08-lucene-phrase-dict)** —— 自訂 Lucene `TokenFilterFactory`（Java SPI），啟動時讀取詞組字典,把被 ICU 過度切分的 CJK token 流重新合回單一詞組 token。中研院 CKIP、HanLP、IK、Jieba user-dict 或內部品牌/品類字典接進 Lucene 分析鏈的範本。

## 起步

```sh
# Vespa CLI
brew install vespa-cli

# Maven（02、03、06、07、08 子例需要）
brew install maven

# Vespa Cloud —— 到 https://cloud.vespa.ai/ 申請免費試用
vespa config set target cloud
vespa config set application TENANT.APP_NAME
vespa auth login
vespa auth cert

# 各子例 README 內附部署與查詢指令
#（通常是 `vespa deploy --add-cert --wait 900 app` 或 `... target/application`）。
# `--add-cert` 會把 `vespa auth cert` 產生的憑證打進應用套件——每個子例
# 第一次部署時需要。
```

迭代結束後，拆掉 dev 部署：

```sh
vespa destroy --force
```

### 改在本機執行

若想離線迭代，每個子例 README 都附 `## 在本機部署 + 測試` 段落，對應的 Docker 流程。該段落同時也是 CI 執行的內容（見下方 [Testing method](#testing-method)）。

[`ext/queries.md`](ext/queries.md) 的對比查詢在八個子例上完全相同——逐個執行，觀察每個子例傳回什麼文件，這就是「哪個 linguistics 選項做什麼事」的答案。每個 schema 也宣告了 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，可以用 `vespa query ... 'summary=debug-tokens'` 直接讀欄位的 token 流，不必登入容器。

## Testing method

每個子例的 `## 在本機部署 + 測試` 段落同時也是可執行的煙霧測試。Vespa sample-apps repo 的 CI 會解析 `<pre>` / `<p>` 上的 `data-test` 屬性，對乾淨的本機 Vespa 容器依序執行。

用到的屬性：

| 屬性 | 意義 |
|---|---|
| `<pre data-test="exec">` | 執行 block 內的 shell 命令。非零 exit code 即視為失敗。 |
| `<pre data-test="exec" data-test-assert-contains="STR">` | 同 `exec`，再加抓到的 stdout 必須含 `STR` 文字。 |

其他 sample app 用的 `init-deploy` macro 這裡不適用 —— 該 macro 寫死 `vespa deploy --wait 300 ./app`，但我們五個子例（`02`、`03`、`06`、`07`、`08`）跑 `mvn package` 後要部署 `target/application`。因此每個子例的第一個 `<pre data-test="exec">` block 都自寫 `git clone … && cd … && docker run … && vespa deploy …`，CI 對 config-only 與 Maven 兩種形態一致處理。

每個子例斷言三件事：

1. **Q1 查詢**（`手机壳`，zh-CN locale）—— 斷言挑*差異性* hit ID：
   - 隔離型（`01`、`02`、`04`、`05`）斷言含 `cn-001`。
   - 跨變體型（`03`、`06`、`07`、`08`）斷言含 `tw-001` —— 證明 Simplified 查詢命中了 Traditional 文件（跨變體訊號）。
2. **Q2 查詢**（`手機殼`，zh-TW locale）—— Q1 的鏡像。
3. **Q6 `debug-tokens` summary** —— 斷言回應裡含某個 token 字串。所選字串反映該子例的 analyzer chain 應該產出什麼：
   - `01`：`手机`（OpenNLP CJK 輸出）
   - `02`：`无线`（SmartCN 詞級 token）
   - `03`：tw-002 的 `title_tokens` 含 `无线`（證明 ICU `Traditional-Simplified` 摺疊，因為源是 `無線`）
   - `04`：`蓝牙`（Jieba 字典詞）
   - `05`：`机壳`（2 字 gram）
   - `06`、`07`：tw-001 的 `title_tokens` 含 `手机壳`（證明 Java 端繁→簡摺疊；源是 `手機殼`）
   - `08`：`空气净化器`（證明 phrase aggregator 把被切碎的 token 收回字典詞組）

訊號是*行為*（查詢命中 ID）+ *內省*（實際在索引裡是什麼）。任一失敗就代表該子例壞掉。

### 本機跑這套測試

```sh
cd 06-opennlp-icu        # 任一子例
# 依序執行 `## 在本機部署 + 測試` 段落內的 block。
# 也可以手動一行一行執行每個 `<pre data-test="exec">` block。
```

要把同一套測試套用到 Vespa Cloud，把 `docker run` + `vespa config set target local` 那段換成 `vespa config set target cloud` + `vespa config set application TENANT.APP_NAME` + `vespa auth login` + `vespa deploy --add-cert --wait 900 ...`。feed 與 query block（含斷言）不需改。

## 術語表

| 術語 | 通俗解釋 |
|---|---|
| **Vespa** | 本 sample app 服務的搜尋引擎。 |
| **Linguistics** | 可插拔的 Vespa 元件，決定文字如何變成可搜尋的 token。兩大類:基於 OpenNLP、基於 Lucene。 |
| **Tokenizer** | 把原始文字切成 token 的元件。CJK tokenizer 多為字典或規則驅動;空白 tokenizer 只按空格切。 |
| **Stemmer** | 把 token 歸約為詞根(英文 `running` → `run`)。對中文意義不大。 |
| **Segmenter** | 對無空格語言而言相當於 tokenizer。Vespa API 區分 Segmenter(查詢側)與 Tokenizer(索引側);實務上同一引擎實作兩者。 |
| **set_language** | Vespa 索引語言指令,給文件欄位標上語言,讓對應 Linguistics 規則生效。每個 schema 在 `language` 欄位用到。 |
| **Match mode** | Schema 欄位選項,決定 token 抽取方式:`text`(預設,走 Linguistics)、`gram`(N 字元重疊窗)、`word`(精確)、`exact`(字面)。 |
| **CJK** | 中、日、韓文,字之間沒有空格。所以斷詞重要。 |
| **簡體中文(zh-CN、zh-Hans)** | 中國大陸與新加坡使用的字符集。`苹果`、`电脑`。 |
| **繁體中文(zh-TW、zh-Hant)** | 台灣使用的字符集;香港有地區變體。`蘋果`、`電腦`。 |
| **zh-HK** | 港式正體。接近 zh-TW 但有港式正字法差異。Vespa 無一等公民支援;當作 zh-TW 用,或用 OpenCC 的 `s2hk` 方向。 |
| **Transliteration(轉寫)** | 把字元從一種字形對應到另一種。`Traditional-Simplified` 是轉寫器;它改字形不改語意。 |
| **Tokenization 與 segmentation** | 同一概念的兩種稱呼。本專案混用。 |
| **n-gram / bigram** | 由 N 個連續字元組成的 token。`bigram` = 2 字元窗。便宜、精度低,不需字典。 |
| **BM25** | 相關性評分公式。我們 schema 中 `first-phase` 排序預設用 BM25。 |
| **OpenNLP** | Apache 的自然語言處理函式庫。Vespa 預設 Linguistics 元件包它。CJK 支援很弱——只有字元 bigram。 |
| **Lucene** | Apache 搜尋函式庫。Vespa 的 `LuceneLinguistics` 讓你以 XML 設定 Lucene 分析鏈。內含數百個 token filter、數十個語言 analyzer。 |
| **ICU** | International Components for Unicode。Unicode 正確性的參考函式庫。Lucene 內含 ICU 綁定(`icuTokenizer`、`icuTransform`)。 |
| **OpenCC** | Open Chinese Convert。專注中文變體轉換,提供地區專屬方向(`s2hk`、`s2twp`、`tw2s` 等)。 |
| **Jieba** | 流行的字典型中文斷詞器。同層 [vespa-chinese-linguistics](../vespa-chinese-linguistics) 將其包裝為 Vespa Linguistics 元件。 |
| **SmartCN** | Lucene 的 HMM 中文斷詞器。基於簡體語料訓練。 |
| **中研院 CKIP** | 中央研究院 CKIP 中文斷詞系統——台灣對繁體中文最強的斷詞器。本專案未直接整合 CKIP,但 08 示範如何把外部字典接進 Lucene。 |
| **HanLP、IK Analyzer** | 其他常見的中文 NLP 函式庫,形態類似 Jieba/CKIP。 |
| **SPI(Service Provider Interface)** | Java 標準機制,透過 `META-INF/services/` 把實作插入函式庫。Lucene 以 SPI 發現 token filter / tokenizer factory。 |
| **CharFilter** | Lucene 分析鏈第一段——斷詞前處理原始文字。 |
| **TokenFilter** | 處理 tokenizer 產出的 token。一條鏈可以串多個；Lucene 多數 filter 都是 TokenFilter。 |
| **TokenizerFactory / TokenFilterFactory** | Lucene 的 SPI 工廠,用來產生實際的 Tokenizer / TokenFilter 實例。自訂 Lucene 擴充實作其中之一。 |
| **OSGi** | Java 用於插件隔離的模組化執行期。Vespa 每個 `container-plugin` JAR 都是一個 OSGi bundle。影響類的跨 bundle 可見性。 |
| **Bundle** | 自含的 Vespa 插件 JAR。Container-plugin Maven 專案產出一個 bundle。 |
| **`getOrig()` 與 `getTokenString()`** | Vespa `Token` 的兩個屬性。`getOrig()` 是欄位的原始子字串;`getTokenString()` 是被索引的形式。自訂 Tokenizer 的變換要在索引時生效,兩者必須不同——見 [06](06-opennlp-icu) 的模式說明。 |
