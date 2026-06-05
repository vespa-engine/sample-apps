<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Linguistics-Asia: choosing a Vespa linguistics option for Chinese

Vespa offers several ways to handle Chinese text — Simplified, Traditional, and the mix you typically see across zh-CN, zh-TW, and zh-HK markets. This directory compares those options side by side. It ships **eight runnable sub-applications** — one per integration shape — over the same dataset and the same five compare-queries, so the trade-offs are observable, not just theoretical.

Each sub-example has its own three-language README (English, 简体中文, 繁體中文). The shared dataset and queries live in [`ext/`](ext/).

## Why tokenization choice dominates Chinese search

Chinese has no whitespace, so the tokenizer's segmentation strategy *is* your matching strategy. A query for `蓝牙耳机` either becomes one token (a dictionary hit), three tokens (`蓝`, `牙`, `耳机` — wrong), or five overlapping bigrams (recall-heavy, precision-light) depending on the analyzer. Simplified and Traditional add another axis: same product, different code points, no recall between them unless you fold scripts.

## How to choose

Two lenses on the same eight examples plus a quick decision table. Pick whichever frame fits how your team is approaching the problem.

### Decision table

| Option | Variants | Word seg | Cross-variant | Needs Maven | Best for |
|---|---|---|---|---|---|
| [01 OpenNLP CJK](01-opennlp-cjk) | zh-CN + zh-TW (bigrams) | no | no | no | smallest working zh-CN app, zero build |
| [02 Lucene per-variant](02-lucene-per-variant) | zh-CN + zh-TW isolated | SmartCN for zh-CN | no | yes | independent zh-CN + zh-TW storefronts |
| [03 Lucene cross-variant](03-lucene-icu) | zh-CN + zh-TW unified | ICU CJK word seg | yes (Lucene `icuTransform`) | yes | one storefront serving both markets, no custom Java |
| [04 Jieba reuse](04-jieba) | zh-CN | Jieba dictionary | no | no (reuses sibling bundle) | zh-CN with custom dictionary support |
| [05 Gram fallback](05-gram-fallback) | any CJK | no | partial (codepoint-shared bigrams only) | no | maximum recall, mixed scripts, no build |
| [06 Custom Java Linguistics](06-opennlp-icu) | zh-CN + zh-TW unified | OpenNLP CJK + per-token ICU fold | yes (Java ICU4J) | yes | template for any custom Java Linguistics |
| [07 OpenCC regional variants](07-opennlp-opencc) | zh-CN + zh-TW + zh-HK unified | OpenNLP CJK + per-token OpenCC4j fold | yes (Java OpenCC4j) | yes | when you need region-aware directions (s2hk, s2twp, hk2s, …) |
| [08 Lucene phrase dictionary](08-lucene-phrase-dict) | zh-CN + zh-TW unified | ICU CJK + custom phrase aggregator | yes (Lucene SPI factory) | yes | when you need a real Chinese phrase dictionary (CKIP, HanLP, IK, brand list) plugged into Lucene |

### Lens A — by mechanism (how you want to integrate)

Four mechanism families, in order of cost:

**1. Stock config (no Java).** Declare a built-in Linguistics in `services.xml`. Pick OpenNLP for the minimum; Lucene when you want richer analyzer chains.
- Smallest possible deploy → **[01](01-opennlp-cjk)** (OpenNLP, character bigrams).
- Word-level segmentation, variants kept separate → **[02](02-lucene-per-variant)** (Lucene).
- Word-level segmentation, variants folded to one index → **[03](03-lucene-icu)** (Lucene).
- Recall over precision, any CJK input → **[05](05-gram-fallback)** (no Linguistics, schema-level `match: gram`).

**2. Reuse a published bundle.** Drop someone else's `container-plugin` JAR into `app/components/`. No Java in your app.
- Jieba dictionary segmentation → **[04](04-jieba)**.

**3. Custom Lucene SPI factory.** Write your own `TokenFilterFactory` / `TokenizerFactory` / `CharFilterFactory`, register through `META-INF/services/`. Stays in the Lucene config path; the factory hosts arbitrary Java logic.
- Phrase dictionary (CKIP, HanLP, IK, brand list, in-house catalogue) → **[08](08-lucene-phrase-dict)**.
- Same shape applies to: DB-backed synonyms, ML-based filters, wrapping OpenCC4j to get `s2hk` / `s2twp` / `hk2s` directions, calling out to a normalization service per token, etc.

**4. Custom OpenNLP-extends Linguistics.** Extend `OpenNlpLinguistics` in Java, override `getTokenizer` / `getStemmer` / `getSegmenter` (and optionally `getDetector`). Exposes the full Vespa Linguistics API surface.
- ICU4J transliterator example → **[06](06-opennlp-icu)**.
- OpenCC4j example (region-aware directions) → **[07](07-opennlp-opencc)**.

Mechanisms 3 and 4 cover roughly the same ground; the choice is structural, not capability-driven. **Prefer 3 (Lucene factory)** unless you specifically need (a) the full Linguistics API including `Detector` / `Segmenter`, (b) tight integration with an existing OpenNLP-extends bundle, or (c) your team's existing investment in the OpenNLP-extends shape. See [The two underlying mechanisms](#the-two-underlying-mechanisms) below for the full trade-off.

### Lens B — by scenario (what your dataset and markets look like)

**Markets covered**
- One market only, zh-CN → **[01-opennlp-cjk](01-opennlp-cjk)** (smallest), **[02-lucene-per-variant](02-lucene-per-variant)** (better segmentation), or **[04-jieba](04-jieba)** (Jieba dictionary if you'll want to extend).
- One market only, zh-TW → **[02-lucene-per-variant](02-lucene-per-variant)**'s zh-TW chain, or **[05-gram-fallback](05-gram-fallback)** if loose recall is acceptable.
- Two markets, kept isolated (independent zh-CN and zh-TW storefronts) → **[02-lucene-per-variant](02-lucene-per-variant)**.
- Two markets, unified search (one storefront serves both zh-CN and zh-TW) → **[03-lucene-icu](03-lucene-icu)** (config) or **[06-opennlp-icu](06-opennlp-icu)** (Java).
- Three markets including zh-HK with region-specific orthography → **[07-opennlp-opencc](07-opennlp-opencc)**, or a Lucene factory wrapping OpenCC4j.
- Multi-script (Chinese + Japanese + Korean + Latin) with recall-first goal → **[05-gram-fallback](05-gram-fallback)**.

**Dataset characteristics**
- Large catalog with brand names, model numbers, industry terms that ICU may over-segment → **[08-lucene-phrase-dict](08-lucene-phrase-dict)** (plug a real Chinese phrase dictionary).
- Lexical regional differences (e.g. 笔记本电脑 vs 笔记型电脑) → any of **03 / 06 / 07 / 08** with a synonym layer added.
- Mixed-case Latin brand names (`iPhone`, `MacBook Pro`) appearing in Chinese text → all examples handle these the same (Latin tokens lowercase normally).

**Team / build constraints**
- No Java skills available, no Maven in CI → **[01-opennlp-cjk](01-opennlp-cjk)**, **[04-jieba](04-jieba)**, or **[05-gram-fallback](05-gram-fallback)** (XML / drop-in JAR only).
- Java team, prefer Lucene ecosystem → **02 / 03 / 08**.
- Java team, prefer OpenNLP-extends shape → **06 / 07**.
- Smallest deployable PoC, will revisit later → **[01-opennlp-cjk](01-opennlp-cjk)**.

### A note on zh-HK (Cantonese)

Vespa has no official zh-HK or yue tokenizer. Three pragmatic paths:

1. **Treat zh-HK as zh-TW** — wire zh-HK to the same chain as 03/06 (both end up Simplified after fold). Works for most ecommerce text since the script difference dominates and HK orthography is close to TW.
2. **For HK-specific orthography**, use OpenCC's `s2hk` / `hk2s` directions. ICU does not ship a `Traditional-HongKong` transliterator, so `icuTransform` alone (03 path) cannot distinguish HK from TW Traditional. Two ways to bring OpenCC's HK directions in:
   - **Java-extension Linguistics** — [07-opennlp-opencc](07-opennlp-opencc) wraps OpenCC4j in an `OpenNlpLinguistics` subclass.
   - **Lucene SPI factory wrapping OpenCC4j** — we don't ship this as a separate sub-example, but it's a straightforward combination of 08's `TokenFilterFactory` pattern with 07's `ZhConverterUtil` call. Same library, Lucene-side integration. Pick this if you want to stay in the Lucene config path elsewhere.
3. **Fall back to gram matching (05)** — codepoint-aware, locale-agnostic, recall over precision.

## The two underlying mechanisms

Once you know which integration path you want (from Lens A above), this section is the deeper read on the choices behind it.

### OpenNLP vs Lucene — the meta-question

Vespa ships two pluggable Linguistics implementations. The choice between them shapes everything else.

| | **OpenNLP-based** (default, `OpenNlpLinguistics`) | **Lucene-based** (`LuceneLinguistics`) |
|---|---|---|
| Out-of-the-box CJK | bigrams only (`cjk=true` config flag) | dictionary word segmentation (ICU, SmartCN, …) |
| Available token filters | minimal | ~100 (synonym, stopword, n-gram, phonetic, ICU, …) |
| Language coverage | English + basic Western, weak elsewhere | ~40 language analyzers shipped |
| Multi-stage analyzer chain | requires Java | declarative XML |
| Per-locale routing | one component, language-agnostic | `<analysis><item key="...">` map |
| Multiple analysis profiles | no | yes — one chain per field or use case (see [Lucene linguistics docs](https://docs.vespa.ai/en/lucene-linguistics.html)) |
| Customization path | extend `OpenNlpLinguistics` in Java (Jieba pattern) | write a Lucene SPI factory (`TokenFilterFactory`, etc.) |
| Maven required | no for default; yes for extensions | yes |
| Bundle size impact | tiny (already loaded) | medium (each `lucene-analysis-*` adds 1-15 MB) |
| Examples in this sample app | 01, 04, 06, 07 | 02, 03, 08 |

**Default to Lucene** unless you specifically need OpenNLP. Reasons:

1. **Chinese-variant work specifically benefits from Lucene's ecosystem.** ICU CJK word segmentation and `Traditional-Simplified` transliteration are config-only with Lucene; with OpenNLP you'd write Java to get the same outcome (see 06).
2. **Multi-language deployments scale better with Lucene.** Adding zh + ja + ko + th means three more `<item>` entries in `services.xml`; with OpenNLP each non-default language would mean another Java extension.
3. **Synonyms, stopwords, dictionary lookups, custom factories — all fit naturally into a Lucene chain.** OpenNLP extensions tend to become single-purpose monoliths because OpenNLP doesn't expose the same chain composition.
4. **The Lucene path is well-documented and battle-tested across the search industry**, not just in Vespa.

OpenNLP is still the right choice when:

- **Stock English + bigram CJK is enough** — you do not need to ship a Maven plugin. 01 is one XML file.
- **You're reusing a published OpenNLP-extends bundle** like [vespa-chinese-linguistics](../vespa-chinese-linguistics) — 04 just drops the JAR into `components/`.
- **You want the full Vespa Linguistics API exposed** — Detector, Stemmer, Segmenter, Tokenizer in one class. The OpenNLP-extends pattern gives you all four override points; Lucene factories give you only the analyzer chain. Examples: 06, 07.

The third point is a *preference*, not a capability gap. Wrapping a third-party library (OpenCC4j, an ML service, a custom dictionary) inside a Lucene `TokenFilterFactory` is also feasible — see 08 for the SPI-factory shape, and the [zh-HK note](#a-note-on-zh-hk-cantonese) for an explicit example. If you've otherwise committed to the Lucene config path, prefer that even for non-stock behavior.

### Customizing via a Lucene SPI factory

For Chinese-variant use cases, stock Lucene SPI (`icuTokenizer`, `icuTransform`, `synonym`, …) already covers most needs — see [02](02-lucene-per-variant) and [03](03-lucene-icu). When you need a filter or tokenizer that doesn't exist as stock SPI, write a Lucene `TokenFilterFactory` / `TokenizerFactory` / `CharFilterFactory`, register via `META-INF/services/`, and reference it by name in `services.xml`.

In this directory, [08-lucene-phrase-dict](08-lucene-phrase-dict) is the worked example — a custom phrase-aggregator filter that reads a dictionary file (CKIP / HanLP / IK / Jieba user-dict / in-house brand list) at startup and combines over-segmented CJK token streams back into single phrase tokens.

For generic Lucene SPI factory patterns unrelated to Chinese (DB-backed synonyms, ML-based filters, phonetic algorithms, etc.), the sibling [lucene-linguistics](../lucene-linguistics) sample app has additional templates: [`add-token-filter-factory`](../lucene-linguistics/add-token-filter-factory) and [`custom-analyzer-phonetic`](../lucene-linguistics/custom-analyzer-phonetic).

### Customizing via an OpenNLP-extends Linguistics

The mechanism is one component declaration in `services.xml`:

```xml
<component id="my.package.MyLinguistics" bundle="my-bundle"/>
```

The class implements [`com.yahoo.language.Linguistics`](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java). In practice you extend `OpenNlpLinguistics` and override `getTokenizer()`, `getStemmer()`, and `getSegmenter()` so all three return the same wrapped tokenizer. [06-opennlp-icu](06-opennlp-icu) is the smallest end-to-end example; [vespa-chinese-linguistics](../vespa-chinese-linguistics) is another working bundle (Jieba).

## The eight sub-examples

All sub-examples share the same `product.sd` schema shape and the same dataset (`ext/documents.jsonl`), so their results are directly comparable when you run the compare queries from `ext/queries.md`.

### Schema requirement: `set_language` field ordering

Every sub-example schema has:

```
field language type string {
  indexing: summary | attribute | set_language
  match: word
}
```

`set_language` tells the indexer which language's Linguistics rules to apply to *subsequent* fields in the document. **Declare the `language` field before any field that needs language-specific tokenization** (here: `title` and `body`). The indexing pipeline processes fields in declaration order; if `language` is declared after `title`/`body`, the `set_language` directive fires too late and the indexer falls back to the container default (English, OpenNLP CJK) regardless of the value in the field. Easy to break, silent at deploy time, only visible as poor recall after feeding.

### Catalog

- **[01-opennlp-cjk](01-opennlp-cjk)** — stock OpenNLP linguistics with the CJK config flag turned on. No code, no Maven, just XML. Bigram-based.
- **[02-lucene-per-variant](02-lucene-per-variant)** — `LuceneLinguistics` with one analyzer chain per locale: SmartCN for zh-CN, Standard + CJK bigram for zh-TW. Word segmentation where it counts; variants stay isolated.
- **[03-lucene-icu](03-lucene-icu)** — Lucene `icuTokenizer` (CJK word seg) + `icuTransform` token filter with `id=Traditional-Simplified`. Both index and query side fold to Simplified, so zh-CN ↔ zh-TW recall works. Pure XML config, no custom Java.
- **[04-jieba](04-jieba)** — pure application package that picks up the prebuilt [`vespa-chinese-linguistics`](../vespa-chinese-linguistics) bundle. Demonstrates the `<component bundle="..."/>` plug-in mechanism with no Java in this sub-example.
- **[05-gram-fallback](05-gram-fallback)** — no Linguistics component at all; `match: gram, gram-size: 2` does the work in-schema. Works for any Unicode input, including mixed scripts.
- **[06-opennlp-icu](06-opennlp-icu)** — custom Java `Linguistics` extending `OpenNlpLinguistics`, wrapping its tokenizer and using ICU4J's `Transliterator` to fold Traditional → Simplified per token. Same outcome as 03, implemented on the Java side. The canonical template for the OpenNLP-extends shape — pick it when you want the full Vespa Linguistics API (Tokenizer / Stemmer / Segmenter / Detector) rather than a Lucene analyzer chain. For the same trad↔simp outcome via Lucene SPI, see 03; for arbitrary Java logic inside a Lucene chain, see 08.
- **[07-opennlp-opencc](07-opennlp-opencc)** — same custom-Java shape as 06 but uses OpenCC4j instead of ICU4J. Reach for this only when you need OpenCC's region-aware directional converters (`s2hk`, `s2twp`, `hk2s`, …) that ICU's single `Traditional-Simplified` doesn't expose. For the generic trad↔simp case, ICU (03 or 06) is cheaper.
- **[08-lucene-phrase-dict](08-lucene-phrase-dict)** — custom Lucene `TokenFilterFactory` (Java SPI) that reads a phrase dictionary at startup and aggregates over-segmented CJK token streams back into single phrase tokens. Template for plugging Academia Sinica CKIP, HanLP, IK, Jieba user-dict, or any in-house brand/catalogue dictionary into a Lucene analyzer chain.

## Getting started

```sh
# Vespa CLI
brew install vespa-cli

# Maven (for sub-examples 02, 03, 06, 07, 08)
brew install maven

# Vespa Cloud — sign up free at https://cloud.vespa-cloud.com/
vespa config set target cloud
vespa config set application TENANT.APP_NAME
vespa auth login
vespa auth cert

# Each sub-example has its own README with deploy + query commands
# (typically `vespa deploy --add-cert --wait 900 app` or `... target/application`).
# `--add-cert` bundles the cert generated by `vespa auth cert` into the application
# package — needed on first deploy of each sub-example.
```

When done iterating, tear the dev deployment down:

```sh
vespa destroy --force
```

### Run locally instead

If you prefer self-managed Vespa for offline iteration, every sub-example README ships a `## Local deploy + test` section with the equivalent Docker flow. That section is also what CI executes (see [Testing method](#testing-method) below).

The compare queries in [`ext/queries.md`](ext/queries.md) are the same across all eight sub-examples — fire them in turn and watch which docs each one returns. That's the answer to "which linguistics option does what." Each schema also declares a `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens), so you can read per-field token streams via `vespa query ... 'summary=debug-tokens'` without shelling into the container.

## Testing method

Every sub-example's `## Local deploy + test` section doubles as an executable smoke test. The Vespa sample-apps repo CI parses `data-test` attributes on `<pre>` / `<p>` blocks and runs them in order against a fresh local Vespa container.

Annotations used:

| Attribute | Meaning |
|---|---|
| `<pre data-test="exec">` | Execute the shell commands inside this block. Fails the test on non-zero exit code. |
| `<pre data-test="exec" data-test-assert-contains="STR">` | Same as `exec`, plus the captured stdout must contain `STR` literally. |

The `init-deploy` macro that other sample apps use cannot be applied here — it hardcodes `vespa deploy --wait 300 ./app`, but five of our sub-examples (`02`, `03`, `06`, `07`, `08`) deploy `target/application` after `mvn package`. Instead each sub-example opens its first `<pre data-test="exec">` block with explicit `git clone … && cd … && docker run … && vespa deploy …` so CI works uniformly for both config-only and Maven shapes.

Per sub-example we assert on three things:

1. **Q1 query** (`手机壳`, zh-CN locale) — assertion picks the *differential* hit ID:
   - Isolated sub-examples (`01`, `02`, `04`, `05`) assert `cn-001` only.
   - Cross-variant sub-examples (`03`, `06`, `07`, `08`) assert `tw-001` — proves the Traditional doc was reached from a Simplified query (the cross-variant signal).
2. **Q2 query** (`手機殼`, zh-TW locale) — mirror of Q1.
3. **Q6 `debug-tokens` summary** — asserts a specific token string is present in the response. The chosen string reflects what each sub-example's analyzer chain is supposed to produce:
   - `01`: `手机` (OpenNLP CJK output)
   - `02`: `无线` (SmartCN word token)
   - `03`: `无线` in tw-002's `title_tokens` (proves ICU `Traditional-Simplified` fold, since source was `無線`)
   - `04`: `蓝牙` (Jieba dictionary word)
   - `05`: `机壳` (2-character gram)
   - `06`, `07`: `手机壳` in tw-001's `title_tokens` (proves Java-side trad→simp fold; source was `手機殼`)
   - `08`: `空气净化器` (proves phrase aggregator collapsed segmented tokens back into the dictionary phrase)

The signal is *behavioral* (query hit IDs) plus *introspective* (what's actually in the index). If either fails, the sub-example is broken.

### Running the tests locally

```sh
cd 06-opennlp-icu        # any sub-example
# Execute the blocks under `## Local deploy + test` in order.
# Or just follow each `<pre data-test="exec">` block by hand.
```

To use the same suite against Vespa Cloud, replace the `docker run` + `vespa config set target local` block with `vespa config set target cloud` + `vespa config set application TENANT.APP_NAME` + `vespa auth login` + `vespa deploy --add-cert --wait 900 ...`. The feed and query blocks (and assertions) work unchanged.

## Glossary

| Term | Plain meaning |
|---|---|
| **Vespa** | The search engine this sample app is for. |
| **Linguistics** | A pluggable Vespa component that decides how text becomes searchable tokens. Two main flavours: OpenNLP-based and Lucene-based. |
| **Tokenizer** | The piece that splits raw text into tokens. CJK tokenizers are dictionary- or rule-based; whitespace tokenizers just split on spaces. |
| **Stemmer** | Reduces tokens to a root form (English `running` → `run`). Mostly irrelevant for Chinese. |
| **Segmenter** | Like a tokenizer for languages without spaces. Vespa's API distinguishes Segmenter (query-side) from Tokenizer (index-side); in practice the same engine implements both. |
| **set_language** | A Vespa indexing-language directive that tags a document's fields with a specific language so the right Linguistics rules apply. We use it on the `language` field in every schema. |
| **Match mode** | Schema field option that picks how tokens are extracted: `text` (default, uses Linguistics), `gram` (n-character overlapping windows), `word` (exact), `exact` (literal). |
| **CJK** | Chinese, Japanese, Korean — written without spaces between words. Why segmentation matters. |
| **Simplified Chinese (zh-CN, zh-Hans)** | Character set used in mainland China and Singapore. `苹果`, `电脑`. |
| **Traditional Chinese (zh-TW, zh-Hant)** | Character set used in Taiwan and (with regional variants) Hong Kong. `蘋果`, `電腦`. |
| **zh-HK** | Hong Kong Traditional. Close to zh-TW but with HK-specific orthography. Vespa has no first-class support; treat as zh-TW or use OpenCC's `s2hk` direction. |
| **Transliteration** | Mapping characters from one script to another. `Traditional-Simplified` is a transliterator; it changes characters, not meaning. |
| **Tokenization vs segmentation** | Different terms for the same idea. We use them interchangeably. |
| **n-gram / bigram** | Tokens formed from N consecutive characters. `bigram` = 2-character windows. Cheap, low precision, no dictionary needed. |
| **BM25** | A relevance-scoring formula. Used as the default `first-phase` ranking expression in our schemas. |
| **OpenNLP** | Apache library for natural-language processing. Vespa's default Linguistics component wraps it. CJK support is minimal — only character bigrams. |
| **Lucene** | Apache search library. Vespa's `LuceneLinguistics` lets you configure Lucene analyzer chains via XML. Comes with hundreds of token filters and dozens of language analyzers. |
| **ICU** | International Components for Unicode. A reference library for Unicode-correct text processing. Lucene includes ICU bindings (`icuTokenizer`, `icuTransform`). |
| **OpenCC** | Open Chinese Convert. A library focused on Chinese variant conversion with region-specific directions (`s2hk`, `s2twp`, `tw2s`, …). |
| **Jieba** | Popular dictionary-based Chinese word segmenter. The sibling [vespa-chinese-linguistics](../vespa-chinese-linguistics) wraps it as a Vespa Linguistics component. |
| **SmartCN** | Lucene's HMM-based Chinese word segmenter. Trained on Simplified. |
| **Academia Sinica CKIP** | 中央研究院 CKIP 中文斷詞系統 — Taiwan's reference Chinese segmenter. Strong on Traditional Chinese. We don't ship CKIP integration, but 08 shows how to plug an external dictionary into Lucene. |
| **HanLP, IK Analyzer** | Other widely-used Chinese NLP libraries, similar in shape to Jieba/CKIP. |
| **SPI (Service Provider Interface)** | Java's standard mechanism for plugging implementations into a library via `META-INF/services/`. Lucene uses SPI to discover token filter / tokenizer factories. |
| **CharFilter** | First stage in a Lucene analyzer chain — processes raw text before tokenization. |
| **TokenFilter** | Processes tokens emitted by the tokenizer. A chain can have many; most Lucene filters are TokenFilters. |
| **TokenizerFactory / TokenFilterFactory** | Lucene's SPI-friendly factories that produce the actual Tokenizer / TokenFilter instances. Custom Lucene extensions implement one of these. |
| **OSGi** | A modular runtime Java uses for plugin isolation. Vespa loads each `container-plugin` JAR as an OSGi bundle. Affects which classes are visible across bundle boundaries. |
| **Bundle** | A self-contained Vespa plugin JAR. Container-plugin Maven projects produce one bundle. |
| **`getOrig()` vs `getTokenString()`** | Two properties on Vespa's `Token`. `getOrig()` is the original substring of the field; `getTokenString()` is the form that gets indexed. They must differ for a custom Tokenizer's transformation to take effect at index time — see [06](06-opennlp-icu) for the pattern. |
