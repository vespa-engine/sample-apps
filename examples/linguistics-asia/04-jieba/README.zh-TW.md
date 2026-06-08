<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

# 04 — 重用既有 Jieba bundle，不寫新 Java

只要 container-plugin bundle 實作了 `com.yahoo.language.Linguistics`，Vespa 就能載入它。鄰近的範例 [vespa-chinese-linguistics](../../vespa-chinese-linguistics) 已經提供了一個——基於 Jieba、包裹 OpenNLP 的斷詞器。本子例展示**消費端**：純應用套件（無 `pom.xml`），直接拿預先建好的 bundle 來用，只在 services.xml 加三行。

適用情境：想要 **Jieba 等級的簡體中文詞級分詞**、支援字典擴充，又不想自己寫或維護 Linguistics 模組。

## 怎麼接

`services.xml` 直接實例化既有 bundle 內的類：

```xml
<component id="com.qihoo.language.JiebaLinguistics"
           bundle="vespa-chinese-linguistics">
  <config name="com.qihoo.language.config.jieba">
    <!-- 可選的字典、停用詞路徑，相對於應用套件根目錄 -->
  </config>
</component>
```

這就是接入自訂 Linguistics 的完整機制：`<component class="..." bundle="..."/>`。`bundle` 要與 JAR 的 bundle-symbolic-name 一致（`vespa-chinese-linguistics`），部署時 JAR 必須放在 `app/components/`。

## 同層 bundle 編譯一次即可

```sh
# 從 sample-apps 根目錄
( cd ../vespa-chinese-linguistics && mvn package )
cp ../vespa-chinese-linguistics/target/vespa-chinese-linguistics-1.0.0-deploy.jar \
   app/components/
```

只有 Jieba bundle 原始碼變更時才需重新編。本應用自身部署不必重新打包它。

## 部署到 Vespa Cloud

到 <https://cloud.vespa.ai/> 申請免費試用，然後：

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-04     # 替換 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

# Jieba 會把「无线蓝牙耳机」切成 ["无线", "蓝牙", "耳机"]
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' \
            'model.locale=zh-CN' 'trace.level=2'
```

`trace.level=2` 可看到查詢端 Jieba 的斷詞結果。索引端用 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) 檢視：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

`title_tokens` 陣列就是 Jieba 切出的詞，例如 `["无线", "蓝牙", "耳机"]`。

## 在本機部署 + 測試

本機 Docker 流程同時作為 CI 煙霧測試。Vespa sample-apps CI 解析下方的 `data-test` 屬性：`<pre data-test="exec">` block 當 shell 執行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整規範見 [parent README — Testing method](../README.zh-TW.md#testing-method)。

從本子例目錄執行：

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/04-jieba
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 vespaengine/vespa
vespa config set target local
vespa status deploy --wait 300
# Build the sibling Jieba bundle once and copy into app/components/.
( cd ../../vespa-chinese-linguistics && mvn -q package )
cp ../../vespa-chinese-linguistics/target/vespa-chinese-linguistics-1.0.0-deploy.jar app/components/
vespa deploy --wait 300 ./app
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

<pre data-test="exec" data-test-assert-contains="蓝牙">
vespa query 'yql=select * from product where id contains "cn-002"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## 能得到什麼、得不到什麼

- **簡體中文分詞**：Jieba 字典對常見複合詞處理較佳——優於 OpenNLP CJK bigram，與 Lucene SmartCN 相當。
- **自訂字典支援**：把 `.dict` 檔放進應用套件，並在 `<config>` 內引用即可。對品牌詞、品類詞很有用。
- **繁體中文**：Jieba 內建字典僅涵蓋簡體。繁體輸入仍會被切，但效果不如簡體。若需繁體支援，可結合 [06-opennlp-icu](../06-opennlp-icu) 的 ICU 摺疊手法，或改用 [03-lucene-icu](../03-lucene-icu)。
- **本目錄毋須維護程式碼**——這裡只有一份純應用套件。

## 參考資料

- 同層範例：[vespa-chinese-linguistics](../../vespa-chinese-linguistics)
- [Linguistics 介面](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java)
- [Jieba (jieba-analysis)](https://github.com/huaban/jieba-analysis)
