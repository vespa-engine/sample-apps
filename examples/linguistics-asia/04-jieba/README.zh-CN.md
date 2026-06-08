<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

# 04 — 复用现成 Jieba bundle，无需新写 Java

只要 container-plugin bundle 实现了 `com.yahoo.language.Linguistics`，Vespa 就能加载它。同级的示例 [vespa-chinese-linguistics](../../vespa-chinese-linguistics) 已经提供了一个——基于 Jieba、包裹 OpenNLP 的分词器。本子例做的是**消费端**：纯应用包（无 `pom.xml`），把预先编好的 bundle 拿来直接用，只在 services.xml 加三行。

适用场景：想要**Jieba 级别的简体中文词级分词**、支持词典扩展，又不想自己写或维护 Linguistics 模块。

## 接入方式

`services.xml` 实例化已有 bundle 里的类：

```xml
<component id="com.qihoo.language.JiebaLinguistics"
           bundle="vespa-chinese-linguistics">
  <config name="com.qihoo.language.config.jieba">
    <!-- 可选的词典、停用词路径，相对应用包根 -->
  </config>
</component>
```

这就是自定义 Linguistics 的完整接入方式：`<component class="..." bundle="..."/>`。`bundle` 要与 JAR 的 bundle-symbolic-name 一致（`vespa-chinese-linguistics`），部署时 JAR 必须放在 `app/components/`。

## 同级 bundle 编一次就行

```sh
# 从 sample-apps 根目录
( cd ../vespa-chinese-linguistics && mvn package )
cp ../vespa-chinese-linguistics/target/vespa-chinese-linguistics-1.0.0-deploy.jar \
   app/components/
```

只有 Jieba bundle 源码改了才需要重新编。本应用本身的部署不需要重打它。

## 部署到 Vespa Cloud

到 <https://cloud.vespa.ai/> 注册免费试用，然后：

```sh
vespa config set target cloud
vespa config set application TENANT.linguistics-asia-04     # 替换 TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 app

vespa feed ../ext/documents.jsonl

# Jieba 会把“无线蓝牙耳机”切成 ["无线", "蓝牙", "耳机"]
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' \
            'model.locale=zh-CN' 'trace.level=2'
```

`trace.level=2` 可看到查询侧 Jieba 的切词结果。索引侧用 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) 查看：

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=3'
```

`title_tokens` 数组里就是 Jieba 切出的词，例如 `["无线", "蓝牙", "耳机"]`。

## 在本机部署 + 测试

本机 Docker 流程同时是 CI 烟雾测试。Vespa sample-apps CI 解析下方的 `data-test` 属性：`<pre data-test="exec">` block 当 shell 执行，`data-test-assert-contains="STR"` 要求 stdout 含 `STR`。完整规范见 [parent README — Testing method](../README.zh-CN.md#testing-method)。

在本子例目录下执行：

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

## 能得到什么、得不到什么

- **简体中文分词**：Jieba 的词典对常见复合词处理较好——优于 OpenNLP CJK bigram，与 Lucene SmartCN 相当。
- **自定义词典支持**：把 `.dict` 文件放进应用包，在 `<config>` 里引用即可。对品牌词、品类词很有用。
- **繁体中文**：Jieba 内置词典只覆盖简体。繁体输入也会被切，但效果差于简体。如果需要繁体支持，可结合 [06-opennlp-icu](../06-opennlp-icu) 的 ICU 折叠思路，或选 [03-lucene-icu](../03-lucene-icu)。
- **本目录无需维护代码**——这里只有一个纯应用包。

## 参考资料

- 同级示例：[vespa-chinese-linguistics](../../vespa-chinese-linguistics)
- [Linguistics 接口](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java)
- [Jieba (jieba-analysis)](https://github.com/huaban/jieba-analysis)
