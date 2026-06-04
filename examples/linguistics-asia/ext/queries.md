<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Compare queries

The same six queries against every sub-example. The point is to see *which docs each sub-example returns* — that's the answer to "which Linguistics option does what."

## Q1 — Simplified term, Simplified-only

```sh
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
```

Expected: hit cn-001.
Cross-variant hit (tw-001) only on cross-variant sub-examples (`03`, `06`, `07`, `08`).

## Q2 — Traditional term, Traditional-only

```sh
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

Expected: hit tw-001.
Cross-variant hit (cn-001) only on cross-variant sub-examples (`03`, `06`, `07`, `08`).

## Q3 — Multi-character segmentation test

```sh
vespa query 'yql=select * from product where default contains "无线蓝牙耳机"' 'model.locale=zh-CN'
```

Segmentation quality varies:
- `01-opennlp-cjk`: bigrams ("无线","线蓝","蓝牙","牙耳","耳机") → hits cn-002.
- `02-lucene-per-variant` (SmartCN): proper words ("无线","蓝牙","耳机") → hits cn-002.
- `03-lucene-icu` (ICU): words → hits cn-002 *and* tw-002.
- `04-jieba`: Jieba words → hits cn-002.
- `05-gram-fallback`: 2-grams across the whole field → loose recall, hits cn-002 and any doc sharing bigrams.
- `06-opennlp-icu`: hits cn-002 *and* tw-002.
- `07-opennlp-opencc`: hits cn-002 *and* tw-002.
- `08-lucene-phrase-dict`: hits cn-002 *and* tw-002 (phrase aggregator + ICU fold).

Inspect tokens with `'trace.level=2'` (query side) or `'summary=debug-tokens'` (index side).

## Q4 — Brand mention

```sh
vespa query 'yql=select * from product where default contains "蘋果"' 'model.locale=zh-TW'
```

Tests Traditional brand recognition. `01`/`02`/`04`/`05` hit only tw-001. `03`/`06`/`07`/`08` hit tw-001 *and* cn-001 ("苹果").

## Q5 — Mixed-script query (rare, but real for ecommerce)

```sh
vespa query 'yql=select * from product where default contains "iPhone"' 'model.locale=zh-CN'
```

All sub-examples should hit both cn-001 and tw-001 — Latin tokens are tokenized identically (and lowercased) on every chain.

## Q6 — Inspect tokens via the `debug-tokens` summary

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

Returns every doc with its `title` / `body` source-text fields *plus* the corresponding `title_tokens` / `body_tokens` arrays — the exact terms written to the index by that sub-example's analyzer chain. Cloud-friendly: no shell access into the container needed. Reference: [Vespa docs — `tokens` summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens).

Expected per sub-example:
- `01-opennlp-cjk`: 2-char gram tokens (`手机`, `机壳` / `手機`, `機殼`).
- `02-lucene-per-variant`: SmartCN word tokens on zh-CN docs; bigrams on zh-TW docs.
- `03-lucene-icu`: ICU-segmented word tokens, **Simplified-only** in `_tokens` regardless of source script.
- `04-jieba`: Jieba word tokens (zh-CN-quality on both variants; zh-TW input gets weaker segmentation).
- `05-gram-fallback`: 2-char grams on every text field.
- `06-opennlp-icu`: parent-tokenizer tokens with ICU `Traditional-Simplified` fold per token — Simplified-only in `_tokens`.
- `07-opennlp-opencc`: same shape as 06 but folded by OpenCC4j `ZhConverterUtil.toSimple` — Simplified-only in `_tokens`.
- `08-lucene-phrase-dict`: ICU tokens with dictionary phrases (e.g. `空气净化器`, `笔记本电脑`) collapsed back into single tokens.

## Cross-variant recall summary

| Query | 01 OpenNLP CJK | 02 Lucene | 03 ICU | 04 Jieba | 05 Gram | 06 ICU-Java | 07 OpenCC | 08 Phrase |
|------|----|----|----|----|----|----|----|----|
| Q1 zh-CN term  | cn only | cn only | **cn+tw** | cn only | cn (+partial) | **cn+tw** | **cn+tw** | **cn+tw** |
| Q2 zh-TW term  | tw only | tw only | **cn+tw** | tw only | tw (+partial) | **cn+tw** | **cn+tw** | **cn+tw** |
| Q3 segmenting  | bigrams | words   | words+cross | words | 2-grams | words+cross | words+cross | phrases+cross |
| Q4 brand       | tw only | tw only | **cn+tw** | tw only | tw (+partial) | **cn+tw** | **cn+tw** | **cn+tw** |
| Q5 Latin       | both    | both    | both    | both    | both          | both        | both        | both |
| Q6 tokens      | grams   | words / bigrams | simp words | jieba words | grams | simp tokens | simp tokens | phrases |
