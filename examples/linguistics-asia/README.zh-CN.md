<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

[English](README.md) | 简体中文 | [繁體中文](README.zh-TW.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Linguistics-Asia：Vespa 中文 linguistics 选型指南

Vespa 处理中文有好几种方式 —— 简体、繁体，以及你在 zh-CN、zh-TW、zh-HK 市场常见的混合情况。本目录把这些选项并排比较。提供**八个可运行的子应用**——每种接入形态一个——共用同一份资料集与同一组对比查询，让取舍是看得见的，不是停留在文档里的。

每个子例都附带三语 README（English、简体中文、繁體中文）。共享数据集与查询位于 [`ext/`](ext/)。

## 为什么分词决定了中文检索

中文不分词，所以 tokenizer 的切分策略**就是**你的匹配策略。查询 `蓝牙耳机` 可能变成一个 token（命中词典）、三个 token（`蓝`、`牙`、`耳机`——错的）、或五个重叠 bigram（召回宽、精度低）——取决于你用哪个分析器。简繁还多一个维度：同一个商品、码点不同，不折叠脚本就没召回。

## 怎么选

针对同样八个范例的两种视角加一张决策表。挑你团队思考问题时贴近的那一边即可。

### 决策表

| 选项 | 变体 | 词级切分 | 跨变体 | 需要 Maven | 适用场景 |
|---|---|---|---|---|---|
| [01 OpenNLP CJK](01-opennlp-cjk) | zh-CN + zh-TW（bigram） | 否 | 否 | 否 | 最小可跑的 zh-CN 应用，零构建 |
| [02 Lucene 每变体](02-lucene-per-variant) | zh-CN + zh-TW 隔离 | SmartCN 处理 zh-CN | 否 | 是 | 独立的 zh-CN 与 zh-TW 店面 |
| [03 Lucene 跨变体](03-lucene-icu) | zh-CN + zh-TW 统一 | ICU CJK 词分词 | 是（Lucene `icuTransform`） | 是 | 一个店面服务两个市场，无自定义 Java |
| [04 Jieba 复用](04-jieba) | zh-CN | Jieba 词典 | 否 | 否（复用同级 bundle） | 需要自定义词典的 zh-CN |
| [05 Gram 兜底](05-gram-fallback) | 任意 CJK | 否 | 部分（仅码点重叠的 bigram） | 否 | 多脚本混合、最大召回、零构建 |
| [06 自定义 Java Linguistics](06-opennlp-icu) | zh-CN + zh-TW 统一 | OpenNLP CJK + 逐 token ICU 折叠 | 是（Java ICU4J） | 是 | 任意自定义 Java Linguistics 的模板 |
| [07 OpenCC 地区变体](07-opennlp-opencc) | zh-CN + zh-TW + zh-HK 统一 | OpenNLP CJK + 逐 token OpenCC4j 折叠 | 是（Java OpenCC4j） | 是 | 需要地区感知方向（s2hk、s2twp、hk2s 等）时选 |
| [08 Lucene 短语字典](08-lucene-phrase-dict) | zh-CN + zh-TW 统一 | ICU CJK + 自定义短语聚合 | 是（Lucene SPI factory） | 是 | 要把真实中文字典（CKIP、HanLP、IK、品牌词表）接进 Lucene 时选 |

### 视角 A —— 按机制（你想怎么接）

四种机制家族，按成本递增：

**1. 纯配置（无 Java）。** 在 `services.xml` 里宣告 Vespa 内建的 Linguistics。最小用 OpenNLP；要丰富的分析链选 Lucene。
- 最小部署 → **[01](01-opennlp-cjk)**（OpenNLP，字符 bigram）。
- 词级分词、变体独立 → **[02](02-lucene-per-variant)**（Lucene）。
- 词级分词、变体折成一个索引 → **[03](03-lucene-icu)**（Lucene）。
- 召回优先、任何 CJK 输入 → **[05](05-gram-fallback)**（无 Linguistics、schema 层 `match: gram`）。

**2. 重用既有 bundle。** 把别人写好的 `container-plugin` JAR 丢进 `app/components/`。本应用零 Java。
- Jieba 字典分词 → **[04](04-jieba)**。

**3. 自定义 Lucene SPI factory。** 自己写 `TokenFilterFactory` / `TokenizerFactory` / `CharFilterFactory`，透过 `META-INF/services/` 注册。继续走 Lucene 配置路径；factory 内部可以跑任意 Java。
- 短语字典（CKIP、HanLP、IK、品牌词表、内部商品词表）→ **[08](08-lucene-phrase-dict)**。
- 同样形状还适用：DB 来源同义词、基于 ML 的 filter、包 OpenCC4j 拿到 `s2hk` / `s2twp` / `hk2s` 方向、按 token 呼叫规范化服务等。

**4. 自定义 OpenNLP-extends Linguistics。** 继承 `OpenNlpLinguistics`，覆盖 `getTokenizer` / `getStemmer` / `getSegmenter`（可选 `getDetector`）。提供 Vespa 完整 Linguistics API。
- ICU4J transliterator 范例 → **[06](06-opennlp-icu)**。
- OpenCC4j 范例（地区感知方向）→ **[07](07-opennlp-opencc)**。

机制 3 与 4 能力大致重叠；选择是结构性而非能力差距。**预设选 3（Lucene factory）**，除非你具体需要（a）完整 Linguistics API 含 `Detector` / `Segmenter`，（b）与既有 OpenNLP-extends bundle 紧密整合，或（c）团队已经投资在 OpenNLP-extends 形状上。详细取舍见下方 [两套底层机制](#两套底层机制) 段。

### 视角 B —— 按场景（你的资料与市场长什么样）

**涵盖的市场**
- 单市场，只 zh-CN → **[01-opennlp-cjk](01-opennlp-cjk)**（最小）、**[02-lucene-per-variant](02-lucene-per-variant)**（更好的分词）、或 **[04-jieba](04-jieba)**（之后想扩 Jieba 词典）。
- 单市场，只 zh-TW → **[02-lucene-per-variant](02-lucene-per-variant)** 的 zh-TW 链，或 **[05-gram-fallback](05-gram-fallback)**（可接受宽召回）。
- 两市场、隔离（独立 zh-CN / zh-TW 店面） → **[02-lucene-per-variant](02-lucene-per-variant)**。
- 两市场、统一搜索（一个店面服务 zh-CN + zh-TW） → **[03-lucene-icu](03-lucene-icu)**（配置）或 **[06-opennlp-icu](06-opennlp-icu)**（Java）。
- 三市场含 zh-HK 港式正字法 → **[07-opennlp-opencc](07-opennlp-opencc)**，或包 OpenCC4j 的 Lucene factory。
- 多脚本混合（中 + 日 + 韩 + 拉丁）召回优先 → **[05-gram-fallback](05-gram-fallback)**。

**资料特征**
- 商品目录大，含品牌名、型号、行业用词，ICU 容易过度切分 → **[08-lucene-phrase-dict](08-lucene-phrase-dict)**（接真实中文短语字典）。
- 地区词汇差（如 笔记本电脑 vs 笔记型电脑） → **03 / 06 / 07 / 08** 任一，配合一层 synonym。
- 混合大小写的拉丁品牌（`iPhone`、`MacBook Pro`）出现在中文文本中 → 所有范例都正常处理（拉丁 token 正常 lowercase）。

**团队 / 建置限制**
- 无 Java 能力、CI 无 Maven → **[01-opennlp-cjk](01-opennlp-cjk)**、**[04-jieba](04-jieba)**、或 **[05-gram-fallback](05-gram-fallback)**（仅 XML / 复制 JAR）。
- Java 团队，偏 Lucene 生态 → **02 / 03 / 08**。
- Java 团队，偏 OpenNLP-extends 形状 → **06 / 07**。
- 最小可部署 PoC，之后再回头扩 → **[01-opennlp-cjk](01-opennlp-cjk)**。

### 关于 zh-HK（粤语）

Vespa 没有官方的 zh-HK 或 yue 分词器。三个务实路径：

1. **把 zh-HK 当成 zh-TW** —— 让 zh-HK 走 03/06 的同一条链（折叠后都是简体）。多数电商文本都能用，因为字形差异是主导，港版繁体与台版接近。
2. **需要港式正字法独立处理**，要用 OpenCC 的 `s2hk` / `hk2s` 方向。ICU 没有 `Traditional-HongKong` 转写器，因此 `icuTransform` 单独（03 路径）无法区分 HK 与 TW 的繁体。把 OpenCC 的 HK 方向接进来有两条路:
   - **Java 扩展 Linguistics** —— [07-opennlp-opencc](07-opennlp-opencc) 把 OpenCC4j 包在 `OpenNlpLinguistics` 子类中。
   - **Lucene SPI factory 包 OpenCC4j** —— 我们没把这做成独立子例，但它就是 08 的 `TokenFilterFactory` 模式 + 07 的 `ZhConverterUtil` 呼叫的组合。同一个 library、不同接入点。若你希望其它部分继续走 Lucene 配置路径，选这条。
3. **回落到 gram 匹配（05）** —— 按码点切，与 locale 无关，召回为先。

## 两套底层机制

知道你要哪种接入路径之后（视角 A 已经给出），本节是这些选择背后的更深说明。

### OpenNLP vs Lucene —— 该用哪一套

Vespa 自带两套可插拔 Linguistics 实现。在这两者之间的选择决定了其它一切。

| | **OpenNLP 系**（默认，`OpenNlpLinguistics`） | **Lucene 系**（`LuceneLinguistics`） |
|---|---|---|
| 开箱即用的 CJK | 只有 bigram（`cjk=true` 旗标） | 字典词分词（ICU、SmartCN 等） |
| 可用的 token filter | 极少 | ~100 个（synonym、stopword、n-gram、phonetic、ICU 等） |
| 语言覆盖 | 英语 + 基本西语，其它较弱 | 内建 ~40 种语言 analyzer |
| 多阶段分析链 | 需要 Java | 声明式 XML |
| 按 locale 路由 | 一个组件，语言无感 | `<analysis><item key="...">` 表 |
| 多套 analysis profile | 否 | 是 —— 每个字段或用途一条链（见 [Lucene linguistics 文档](https://docs.vespa.ai/en/lucene-linguistics.html)） |
| 自定义路径 | 用 Java 继承 `OpenNlpLinguistics`（Jieba 模式） | 写 Lucene SPI factory（`TokenFilterFactory` 等） |
| 是否需要 Maven | 默认不需要；扩展需要 | 需要 |
| Bundle 大小影响 | 极小（已加载） | 中等（每个 `lucene-analysis-*` 多 1-15 MB） |
| 本项目的子例 | 01、04、06、07 | 02、03、08 |

**默认选 Lucene**，除非你有具体理由用 OpenNLP。原因:

1. **中文变体场景特别受益于 Lucene 生态**。ICU 的 CJK 词分词与 `Traditional-Simplified` 转写在 Lucene 是纯配置；用 OpenNLP 要达到同效果就得写 Java（见 06）。
2. **多语言部署用 Lucene 扩展更好**。要支援 zh + ja + ko + th，只是在 `services.xml` 多写三条 `<item>`；OpenNLP 每个非预设语言都意味多写一个 Java 扩展。
3. **同义词、停用词、字典查找、自订 factory —— 都自然嵌入 Lucene 链**。OpenNLP 扩展容易变成单一用途的整块代码，因为 OpenNLP 没暴露同等的链组合 API。
4. **Lucene 路径文件齐、产业里被验证过**，不只是在 Vespa 内。

OpenNLP 才是正确选择的情况：

- **内建英语 + bigram CJK 已经够** —— 不需要发布 Maven 插件。01 就是一个 XML。
- **复用已发布的 OpenNLP-extends bundle**，例如 [vespa-chinese-linguistics](../vespa-chinese-linguistics) —— 04 就是把 JAR 丢进 `components/`。
- **想要 Vespa 完整 Linguistics API 一次暴露** —— Detector、Stemmer、Segmenter、Tokenizer 都在一个类里。OpenNLP-extends 模式提供四个覆盖点；Lucene factory 只暴露 analyzer 链。例:06、07。

第三点是**偏好**，不是能力差距。把第三方 library（OpenCC4j、ML 服务、自定义字典）包进 Lucene `TokenFilterFactory` 同样可行 —— 见 08 的 SPI factory 形状，以及 [关于 zh-HK 段](#关于-zh-hk粤语) 的具体说明。若你其它部分已经选定 Lucene 配置路径，即使是非内建行为也优先用 Lucene。

### 通过 Lucene SPI factory 自定义

中文变体场景下，内建 Lucene SPI（`icuTokenizer`、`icuTransform`、`synonym` 等）已能覆盖大多需求——见 [02](02-lucene-per-variant) 与 [03](03-lucene-icu)。若需要一个不存在的 filter 或 tokenizer，就写一个 Lucene `TokenFilterFactory` / `TokenizerFactory` / `CharFilterFactory`，透过 `META-INF/services/` 注册，然后在 `services.xml` 用名字引用。

本目录内 [08-lucene-phrase-dict](08-lucene-phrase-dict) 是实战范例 —— 自定义的短语聚合 filter，启动时读取字典文件（CKIP / HanLP / IK / Jieba user-dict / 内部品牌词表），把被过度切分的 CJK token 流重新合回单一短语 token。

非中文相关的通用 Lucene SPI factory 模式（DB 来源同义词、基于 ML 的 filter、phonetic 算法等），同级的 [lucene-linguistics](../lucene-linguistics) 范例已有额外模板：[`add-token-filter-factory`](../lucene-linguistics/add-token-filter-factory) 与 [`custom-analyzer-phonetic`](../lucene-linguistics/custom-analyzer-phonetic)。

### 通过 OpenNLP-extends Linguistics 自定义

机制就是 `services.xml` 里一条 component 声明：

```xml
<component id="my.package.MyLinguistics" bundle="my-bundle"/>
```

类实现 [`com.yahoo.language.Linguistics`](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java)。实践中继承 `OpenNlpLinguistics` 并覆盖 `getTokenizer()`、`getStemmer()`、`getSegmenter()`，三者返回同一个包装后的 tokenizer。[06-opennlp-icu](06-opennlp-icu) 是最小的端到端示例；[vespa-chinese-linguistics](../vespa-chinese-linguistics) 是另一份可工作的 bundle（Jieba）。

## 八个子例

所有子例共用同一份 `product.sd` schema 形状与同一份数据集（`ext/documents.jsonl`），结果可以直接对比——跑 `ext/queries.md` 的对比查询即可。

### Schema 要点：`set_language` 字段顺序

每个子例 schema 都有：

```
field language type string {
  indexing: summary | attribute | set_language
  match: word
}
```

`set_language` 告诉索引器对文档中**后续**字段应用哪种语言的 Linguistics 规则。**`language` 字段必须在任何需要按语言切词的字段之前宣告**（这里是 `title` 与 `body`）。indexing pipeline 按宣告顺序处理字段；若 `language` 放在 `title`/`body` 之后，`set_language` 触发时已经太晚，索引器回退到容器默认（English，OpenNLP CJK），不论字段值是什么。容易破，部署时没警告，喂数据后才以「召回差」表现出来。

### 目录

- **[01-opennlp-cjk](01-opennlp-cjk)** —— 内建 OpenNLP linguistics，开启 CJK 配置项。无代码、无 Maven，仅 XML。基于 bigram。
- **[02-lucene-per-variant](02-lucene-per-variant)** —— `LuceneLinguistics`，每个 locale 一套分析链：zh-CN 用 SmartCN，zh-TW 用 Standard + CJK bigram。该上词级分词的地方上词级；变体保持隔离。
- **[03-lucene-icu](03-lucene-icu)** —— Lucene `icuTokenizer`（CJK 词分词）+ 带 `id=Traditional-Simplified` 的 `icuTransform` token filter。索引与查询两侧都折成简体，所以 zh-CN ↔ zh-TW 互相召回。纯 XML 配置，无自定义 Java。
- **[04-jieba](04-jieba)** —— 纯应用包，直接复用预先编好的 [`vespa-chinese-linguistics`](../vespa-chinese-linguistics) bundle。演示 `<component bundle="..."/>` 接入方式，本子例里没有 Java 代码。
- **[05-gram-fallback](05-gram-fallback)** —— 完全不声明 Linguistics 组件；`match: gram, gram-size: 2` 在 schema 里做完所有事。任何 Unicode 输入（包括多脚本混合）都能跑。
- **[06-opennlp-icu](06-opennlp-icu)** —— 自定义 Java `Linguistics`，继承 `OpenNlpLinguistics`，包装其 tokenizer，用 ICU4J 的 `Transliterator` 逐 token 做繁→简折叠。效果与 03 等价，在 Java 层做。OpenNLP-extends 形态的标准模板——当你想要 Vespa 完整 Linguistics API（Tokenizer / Stemmer / Segmenter / Detector）而不是 Lucene 分析链时用。同样繁简效果但走 Lucene SPI 请看 03；任意 Java 逻辑塞进 Lucene 链请看 08。
- **[07-opennlp-opencc](07-opennlp-opencc)** —— 与 06 同样的自定义 Java 形状但用 OpenCC4j 取代 ICU4J。仅在你需要 OpenCC 的地区感知方向（`s2hk`、`s2twp`、`hk2s` 等）时再用——ICU 单一的 `Traditional-Simplified` 不暴露这层。一般繁简场景 ICU（03 或 06）成本更低。
- **[08-lucene-phrase-dict](08-lucene-phrase-dict)** —— 自定义 Lucene `TokenFilterFactory`（Java SPI），启动时读取短语字典，把被 ICU 过度切分的 CJK token 流重新合回单一短语 token。中研院 CKIP、HanLP、IK、Jieba user-dict 或内部品牌/类目字典接进 Lucene 分析链的模板。

## 起步

```sh
# Vespa CLI
brew install vespa-cli

# Maven（02、03、06、07、08 子例需要）
brew install maven

# Vespa Cloud —— 到 https://cloud.vespa.ai/ 注册免费试用
vespa config set target cloud
vespa config set application TENANT.APP_NAME
vespa auth login
vespa auth cert

# 各子例的 README 内都附部署与查询命令
#（通常是 `vespa deploy --add-cert --wait 900 app` 或 `... target/application`）。
# `--add-cert` 会把 `vespa auth cert` 产生的凭证打进应用包——每个子例
# 第一次部署时需要。
```

迭代结束后，拆掉 dev 部署：

```sh
vespa destroy --force
```

### 改在本机执行

如果想离线迭代，每个子例 README 都附 `## 在本机部署 + 测试` 段落，对应的 Docker 流程。该段落也正是 CI 跑的内容（见下方 [Testing method](#testing-method)）。

[`ext/queries.md`](ext/queries.md) 的对比查询在八个子例上是相同的——逐个跑过去，观察每个子例返回什么文档，这就是「哪个 linguistics 选项做了什么」的答案。每个 schema 也都宣告了 `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens)，可以用 `vespa query ... 'summary=debug-tokens'` 直接读字段的 token 流，不必进容器。

## Testing method

每个子例的 `## 在本机部署 + 测试` 段落同时是可执行的烟雾测试。Vespa sample-apps 仓库的 CI 会解析 `<pre>` / `<p>` 上的 `data-test` 属性，对干净的本机 Vespa 容器按顺序执行。

用到的属性：

| 属性 | 含义 |
|---|---|
| `<pre data-test="exec">` | 执行 block 内的 shell 命令。非零退出码即视为失败。 |
| `<pre data-test="exec" data-test-assert-contains="STR">` | 同 `exec`，再加上抓到的 stdout 必须包含 `STR` 文本。 |

其它 sample app 用的 `init-deploy` macro 在这里不适用 —— 该 macro 写死 `vespa deploy --wait 300 ./app`，但我们五个子例（`02`、`03`、`06`、`07`、`08`）跑 `mvn package` 后要部署 `target/application`。所以每个子例的第一个 `<pre data-test="exec">` block 都自己写 `git clone … && cd … && docker run … && vespa deploy …`，让 CI 对 config-only 与 Maven 两种形态一致处理。

每个子例断言三件事：

1. **Q1 查询**（`手机壳`，zh-CN locale）—— 断言挑*差异性* hit ID：
   - 隔离型（`01`、`02`、`04`、`05`）断言含 `cn-001`。
   - 跨变体型（`03`、`06`、`07`、`08`）断言含 `tw-001` —— 证明 Simplified 查询命中了 Traditional 文档（跨变体信号）。
2. **Q2 查询**（`手機殼`，zh-TW locale）—— Q1 的镜像。
3. **Q6 `debug-tokens` summary** —— 断言响应里包含某个 token 字符串。所选字符串反映该子例的 analyzer chain 应该产生什么：
   - `01`：`手机`（OpenNLP CJK 输出）
   - `02`：`无线`（SmartCN 词级 token）
   - `03`：tw-002 的 `title_tokens` 含 `无线`（证明 ICU `Traditional-Simplified` 折叠，因为源是 `無線`）
   - `04`：`蓝牙`（Jieba 词典词）
   - `05`：`机壳`（2 字 gram）
   - `06`、`07`：tw-001 的 `title_tokens` 含 `手机壳`（证明 Java 端繁→简折叠；源是 `手機殼`）
   - `08`：`空气净化器`（证明 phrase aggregator 把被切碎的 token 收回字典短语）

信号是*行为*（查询命中 ID）+ *内省*（实际在索引里是什么）。任一失败就代表该子例坏掉。

### 本地跑这套测试

```sh
cd 06-opennlp-icu        # 任一子例
# 依序执行 `## 在本机部署 + 测试` 段落内的 block。
# 也可以手动一行一行执行每个 `<pre data-test="exec">` block。
```

要把同一套测试用在 Vespa Cloud 上，把 `docker run` + `vespa config set target local` 那段换成 `vespa config set target cloud` + `vespa config set application TENANT.APP_NAME` + `vespa auth login` + `vespa deploy --add-cert --wait 900 ...`。feed 与 query block（含断言）不需改。

## 术语表

| 术语 | 通俗解释 |
|---|---|
| **Vespa** | 本 sample app 服务的搜索引擎。 |
| **Linguistics** | 可插拔的 Vespa 组件，决定文本如何变成可搜索的 token。两大类：基于 OpenNLP、基于 Lucene。 |
| **Tokenizer** | 把原始文本切成 token 的组件。CJK tokenizer 多为字典或规则驱动；空白 tokenizer 只按空格切。 |
| **Stemmer** | 把 token 归约为词根（英文 `running` → `run`）。对中文意义不大。 |
| **Segmenter** | 对无空格语言而言相当于 tokenizer。Vespa API 区分 Segmenter（查询侧）与 Tokenizer（索引侧）；实务上同一个引擎实现两者。 |
| **set_language** | Vespa 索引语言指令，给文档字段标上语言，让对应 Linguistics 规则生效。每个 schema 在 `language` 字段用到。 |
| **Match mode** | Schema 字段选项，决定 token 抽取方式：`text`（默认，走 Linguistics）、`gram`（N 字符重叠窗）、`word`（精确）、`exact`（字面量）。 |
| **CJK** | 中、日、韩文，没有词间空白。所以分词重要。 |
| **简体中文（zh-CN、zh-Hans）** | 中国大陆与新加坡使用的字符集。`苹果`、`电脑`。 |
| **繁体中文（zh-TW、zh-Hant）** | 台湾使用的字符集；香港有地区变体。`蘋果`、`電腦`。 |
| **zh-HK** | 港式正体。接近 zh-TW 但有港式正字法差异。Vespa 无一等公民支持；当作 zh-TW 用，或用 OpenCC 的 `s2hk` 方向。 |
| **Transliteration（转写）** | 把字符从一种字形映射到另一种。`Traditional-Simplified` 是转写器；它改字形不改语义。 |
| **Tokenization 与 segmentation** | 同一概念的两种叫法。本项目混用。 |
| **n-gram / bigram** | 由 N 个连续字符组成的 token。`bigram` = 2 字符窗口。便宜、精度低，不需要字典。 |
| **BM25** | 相关性评分公式。我们 schema 中 `first-phase` 排序默认用 BM25。 |
| **OpenNLP** | Apache 的自然语言处理库。Vespa 默认 Linguistics 组件包装它。CJK 支持很弱——只有字符 bigram。 |
| **Lucene** | Apache 搜索库。Vespa 的 `LuceneLinguistics` 让你通过 XML 配置 Lucene 分析链。内建几百个 token filter、几十个语言 analyzer。 |
| **ICU** | International Components for Unicode。Unicode 正确性的参考库。Lucene 内含 ICU 绑定（`icuTokenizer`、`icuTransform`）。 |
| **OpenCC** | Open Chinese Convert。专注中文变体转换，提供地区专属方向（`s2hk`、`s2twp`、`tw2s` 等）。 |
| **Jieba** | 流行的字典型中文分词器。同级 [vespa-chinese-linguistics](../vespa-chinese-linguistics) 把它包成 Vespa Linguistics 组件。 |
| **SmartCN** | Lucene 的 HMM 中文分词器。基于简体语料训练。 |
| **中研院 CKIP** | 中央研究院 CKIP 中文断词系统——台湾对繁体中文最强的分词器。本项目不直接整合 CKIP，但 08 演示如何把外部字典接进 Lucene。 |
| **HanLP、IK Analyzer** | 其它常用的中文 NLP 库，形态与 Jieba/CKIP 类似。 |
| **SPI（Service Provider Interface）** | Java 标准机制，通过 `META-INF/services/` 把实现插入到库。Lucene 用 SPI 发现 token filter / tokenizer factory。 |
| **CharFilter** | Lucene 分析链的首段——分词前处理原始文本。 |
| **TokenFilter** | 处理 tokenizer 产出的 token。一条链可以串多个；Lucene 多数 filter 都是 TokenFilter。 |
| **TokenizerFactory / TokenFilterFactory** | Lucene 的 SPI 工厂，用来产生实际的 Tokenizer / TokenFilter 实例。自定义 Lucene 扩展实现其中之一。 |
| **OSGi** | Java 用于插件隔离的模块化运行时。Vespa 每个 `container-plugin` JAR 都是一个 OSGi bundle。影响类的跨 bundle 可见性。 |
| **Bundle** | 一个自包含的 Vespa 插件 JAR。Container-plugin Maven 项目产出一个 bundle。 |
| **`getOrig()` 与 `getTokenString()`** | Vespa `Token` 的两个属性。`getOrig()` 是字段中原始子串；`getTokenString()` 是被索引的形式。自定义 Tokenizer 的变换要在索引时生效，两者必须不同——见 [06](06-opennlp-icu) 的模式说明。 |
