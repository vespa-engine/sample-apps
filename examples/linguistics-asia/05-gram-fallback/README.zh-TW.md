<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | [简体中文](README.zh-CN.md) | 繁體中文

# 05 — Gram 兜底方案，不用任何 linguistics 元件

有時候正解就是：別在斷詞上動腦筋。Vespa schema 提供 `match: gram` 模式，索引時與查詢時都把欄位切成固定長度的重疊字元 n-gram。不需要 `Linguistics` 元件、不需要字典、不需要斷詞模型。任何 CJK 輸入都能用，任何 Unicode 輸入也都能用，因為它只看字元。

適用情境：
- 想要**最大召回**，可以接受些許過度匹配。
- 不能或不想發布 Java bundle。
- 索引混雜多種變體或腳本（zh-CN、zh-TW、日文漢字等），單一規則無法涵蓋。

## 這個 app 做了什麼

`services.xml` 沒有宣告任何 `Linguistics` 元件。Vespa 仍會載入內建 OpenNLP，但 gram 匹配欄位會繞過它，全由 schema 的 match 規則切。

`app/schemas/product.sd` 的關鍵段落：

```
field title type string {
  indexing: index | summary
  index: enable-bm25
  match {
    gram
    gram-size: 2
  }
}
```

「無線藍牙耳機」會被切成 bigram：`無線`、`線藍`、`藍牙`、`牙耳`、`耳機`。任何含相應 bigram 的文件都會被召回。BM25 排序照常運作——倒排索引不在乎一個 token 是「詞」還是 2-gram。

## 部署到 Vespa Cloud

到 <https://cloud.vespa.ai/> 申請免費試用，然後：

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-05     # 替換 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

預期：每個查詢命中對應變體的文件，且因為 bigram 重疊有機會串到另一變體。這是它的取捨。

## 檢視分詞結果

schema 宣告了 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，查詢即可看到實際寫入索引的 2-gram：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

`title_tokens` 陣列會顯示重疊 2-gram（簡體如 `手机`、`机壳`；繁體如 `手機`、`機殼`）。

## 在本機部署 + 測試

本機 Docker 流程同時作為 CI 煙霧測試。Vespa sample-apps CI 解析下方的 `data-test` 屬性：`<pre data-test="exec">` block 當 shell 執行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整規範見 [parent README — Testing method](../README.zh-TW.md#testing-method)。

從本子例目錄執行：

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/05-gram-fallback
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

<pre data-test="exec" data-test-assert-contains="机壳">
vespa query 'yql=select * from product where id contains "cn-001"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## 取捨

- **索引會變大**：每個字元大約一筆 posting。小型目錄沒問題（本例筆電可跑）；大型目錄要做容量規劃。
- **沒有詞幹化、停用詞、同義詞**：bigram 上的 BM25 比較粗，但對小型 zh-TW 商品目錄有時勝過把品名切錯的「聰明」斷詞器。
- **變體串流**：簡繁共用的 bigram 會跨匹配；但碼點不同的 bigram（如 `机` 對 `機`）仍各自獨立。所以 05 只是半個繁簡互通方案。
- **拉丁字母不受影響**：gram 規則只切漢字，連續的拉丁字母仍作整體 token。

若想要更收斂的召回語意，可參考 [01-opennlp-cjk](../01-opennlp-cjk)（同樣思路，但非 CJK 文字會走完整 OpenNLP 管線），或升級到 [02-lucene-per-variant](../02-lucene-per-variant)（詞級分詞）。

## 參考資料

- [Vespa text matching — gram](https://docs.vespa.ai/en/querying/text-matching.html#gram)
- [Schema reference — `match`](https://docs.vespa.ai/en/reference/schema-reference.html#match)
