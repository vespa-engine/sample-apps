<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

# 01 — OpenNLP 啟用 CJK 分詞

Vespa 容器內建 `OpenNlpLinguistics`，預設不處理中日韓文。開啟兩個 config 旗標後，中文會以相鄰 2 字元切成 bigram。不需寫 Java、不需 Maven，只用 XML 設定。

適用情境：zh-TW 商品目錄不大、對分詞品質要求有限時，最便宜的起步方案。

## 這個 app 做了什麼

- `services.xml` 宣告 `OpenNlpLinguistics`，並設定 `<cjk>true</cjk>` 與 `<createCjkGrams>true</createCjkGrams>`。
- 中文文字被切成相鄰 2 字的 gram：`無線藍牙耳機` → `無線`、`線藍`、`藍牙`、`牙耳`、`耳機`。查詢字也做相同處理，BM25 可正常運作。
- 拉丁文沿用 OpenNLP 預設分詞。

## 限制

- **不區分字形腳本**：簡體與繁體都走相同的 bigram，只看 Unicode 碼點。zh-TW 查詢 `手機殼` 不會命中 zh-CN 文件中的 `手机壳`。
- **沒有詞幹化、沒有停用詞、沒有同義詞**：只做切位置。
- **索引會變大**：每段 N 字的文字會產生約 N 筆 posting。

若以上任一點對你不可接受，請參考 [02-lucene-per-variant](../02-lucene-per-variant)（詞級分詞）、[04-jieba](../04-jieba)（Jieba 字典分詞），或 [06-opennlp-icu](../06-opennlp-icu)（繁簡互通召回）。

## 部署到 Vespa Cloud

到 <https://cloud.vespa-cloud.com/> 申請免費試用，然後：

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-01     # 替換 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

# 查詢 —— 繁體詞只命中繁體文件
vespa query 'yql=select * from product where default contains "手機殼"' \
            'model.locale=zh-TW'

# 同款的簡體文件 —— 這裡查不到
vespa query 'yql=select * from product where default contains "手机壳"' \
            'model.locale=zh-CN'
```

完整對照查詢請見 [../ext/queries.md](../ext/queries.md)。

## 檢視分詞結果

schema 宣告了 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，逐欄位暴露 token 流。查詢它、讀回應裡的 `title_tokens` / `body_tokens`：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

繁體文件會看到 `手機`、`機殼` 之類 2-gram；簡體則是 `手机`、`机壳` —— 碼點不同、bigram 不同，不互相命中。

## 在本機部署 + 測試

本機 Docker 流程同時作為 CI 煙霧測試。Vespa sample-apps CI 解析下方的 `data-test` 屬性：`<pre data-test="exec">` block 當 shell 執行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整規範見 [parent README — Testing method](../README.zh-TW.md#testing-method)。

從本子例目錄執行：

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/01-opennlp-cjk
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 vespaengine/vespa
vespa config set target local
vespa status deploy --wait 300
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

<pre data-test="exec" data-test-assert-contains="手机">
vespa query 'yql=select * from product where id contains "cn-001"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## 參考資料

- [Vespa OpenNLP linguistics](https://docs.vespa.ai/en/linguistics/linguistics-opennlp.html)
- [Vespa 電子報 — OpenNLP CJK 支援](https://blog.vespa.ai/vespa-newsletter-august-2024/)
