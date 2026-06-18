<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

English | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

# 06 — Custom Java Linguistics (template + ICU fold)

A custom `Linguistics` component in ~70 lines of Java. Extends Vespa's `OpenNlpLinguistics`, wraps the inherited `Tokenizer`, and on each token sets the `tokenString` to the ICU `Traditional-Simplified` transliteration of the token's original text. zh-CN ↔ zh-TW recall works at both index and query time.

This is the **template you copy** when you need a custom Linguistics in Vespa — the structure (extend `OpenNlpLinguistics`, override `getTokenizer`/`getStemmer`/`getSegmenter`, wrap the parent's tokenizer) is the same regardless of what your transformation does. The ICU trad↔simp fold here is just an illustrative payload.

If you only need cross-variant recall and don't need a custom Java bundle, [03-lucene-icu](../03-lucene-icu) does the same outcome in pure XML config (Lucene `icuTransform`). Pick 06 when you specifically want the **OpenNLP-extends shape** — i.e. exposing the full Vespa Linguistics API (`Tokenizer` / `Stemmer` / `Segmenter` / `Detector`) at one class instead of a Lucene analyzer chain. Most arbitrary Java logic — brand-name protection, custom dictionary lookups, calling a model server per token — can equally be put inside a Lucene `TokenFilterFactory` (see [08](../08-lucene-phrase-dict) for that shape); the choice between 06 and 08 is mostly about which Vespa surface you prefer to extend.

## The two classes

[`IcuLinguistics.java`](src/main/java/ai/vespa/examples/linguistics/asia/IcuLinguistics.java):

```java
public class IcuLinguistics extends OpenNlpLinguistics {
    private final Tokenizer tokenizer;

    public IcuLinguistics() {
        super();
        this.tokenizer = new IcuTokenizer(super.getTokenizer());
    }

    @Override public Tokenizer getTokenizer()  { return tokenizer; }
    @Override public Stemmer   getStemmer()    { return new StemmerImpl(tokenizer); }
    @Override public Segmenter getSegmenter()  { return new SegmenterImpl(tokenizer); }
}
```

Three overrides, all returning the *same* wrapped tokenizer instance. `getTokenizer` is used by the index-time pipeline; `getStemmer` / `getSegmenter` by the query side. Keep them in sync — if one returns the parent's raw tokenizer, that code path bypasses your wrapping.

### Different behavior at index vs query

If you need *different* tokenization on the two sides (e.g. wider segmentation at index for better recall, tighter at query, or vice versa), build two `Tokenizer` instances and route them separately:

```java
public IcuLinguistics() {
    this.indexTokenizer = new IcuTokenizer(super.getTokenizer(), Mode.INDEX);
    this.queryTokenizer = new IcuTokenizer(super.getTokenizer(), Mode.SEARCH);
}

@Override public Tokenizer getTokenizer() { return indexTokenizer; }
@Override public Stemmer   getStemmer()   { return new StemmerImpl(queryTokenizer); }
@Override public Segmenter getSegmenter() { return new SegmenterImpl(queryTokenizer); }
```

This is the pattern [`JiebaLinguistics`](../../vespa-chinese-linguistics/src/main/java/com/qihoo/language/JiebaLinguistics.java) uses, where Jieba's `SegMode.INDEX` produces broader, more-overlapping segmentation while `SegMode.SEARCH` produces tighter segmentation for query parsing.

For OpenCC / ICU fold the two sides do exactly the same thing, so we use one tokenizer instance. Reach for the two-mode shape only when index and query genuinely need to diverge.

[`IcuTokenizer.java`](src/main/java/ai/vespa/examples/linguistics/asia/IcuTokenizer.java):

```java
public class IcuTokenizer implements Tokenizer {
    private static final Transliterator T2S = Transliterator.getInstance("Traditional-Simplified");
    private final Tokenizer delegate;

    public IcuTokenizer(Tokenizer delegate) { this.delegate = delegate; }

    private Iterable<Token> wrap(Iterable<Token> raw) {
        List<Token> out = new ArrayList<>();
        for (Token t : raw) {
            String origText = t.getOrig();
            SimpleToken st = new SimpleToken(origText)
                    .setOffset(t.getOffset())
                    .setType(t.getType())
                    .setTokenString(T2S.transliterate(origText));
            out.add(st);
        }
        return out;
    }

    @Override public Iterable<Token> tokenize(String input, LinguisticsParameters parameters) {
        return wrap(delegate.tokenize(input, parameters));
    }

    @Override @SuppressWarnings("deprecation")
    public Iterable<Token> tokenize(String input, Language language, StemMode stemMode, boolean removeAccents) {
        return wrap(delegate.tokenize(input, language, stemMode, removeAccents));
    }
}
```

`services.xml`:

```xml
<component id="ai.vespa.examples.linguistics.asia.IcuLinguistics"
           bundle="linguistics-asia-icu"/>
```

## The pattern that matters: per-token `tokenString`, not pre-normalization of input

Tokenize the **original** input, then per-token set `tokenString` to your transformed value. Do **not** normalize the input string before passing it to the delegate. Each token must have:

| | Set to | Why |
|---|---|---|
| `getOrig()` | original substring (Traditional `蘋果`) | Vespa uses this as the source-text reference for the token's offset+length |
| `getTokenString()` | transformed form (Simplified `苹果`) | Indexed and matched against — this is the "search key" |

The two must **differ** for the transformation to take effect at index time. When they are equal, Vespa stores the indexed form as `null` (an "no override" optimization that means "use the original field substring") — which silently undoes any input-level normalization you may have done. The pattern above keeps them distinct.

[`JiebaTokenizer`](../../vespa-chinese-linguistics/src/main/java/com/qihoo/language/JiebaTokenizer.java) uses the same shape: `new SimpleToken(originToken).setTokenString(segmentedWord)`.

## Build, test, deploy to Vespa Cloud

Sign up at <https://cloud.vespa.ai/>, then:

```sh
mvn test
mvn clean package

vespa config set target cloud
vespa config set application TENANT.linguistics-asia-06     # replace TENANT
vespa auth login
vespa auth cert
vespa deploy --add-cert --wait 900 target/application

vespa feed ../ext/documents.jsonl

# Cross-variant queries — both should hit cn-001 AND tw-001
vespa query 'yql=select * from product where default contains "手机壳"' 'model.locale=zh-CN'
vespa query 'yql=select * from product where default contains "手機殼"' 'model.locale=zh-TW'
```

## Verify the index contains Simplified terms only

Query the `debug-tokens` [document-summary](https://docs.vespa.ai/en/reference/schemas/schemas.html#tokens) and read `title_tokens` / `body_tokens` in the response:

```sh
vespa query 'yql=select * from product where true' \
            'summary=debug-tokens' 'hits=20'
```

Expected: every `title_tokens` array contains only Simplified characters (e.g. `手机壳`, `苹果`), even for documents whose `title` summary field still reads Traditional. The ICU fold runs at index time on the token stream — the source-text summary is untouched.

## Local deploy + test

The local Docker flow doubles as the CI smoke test. The Vespa sample-apps CI parses the `data-test` attributes below: `&lt;pre data-test="exec"&gt;` blocks run as shell, and `data-test-assert-contains="STR"` requires `STR` in the captured stdout. The Full conventions in [parent README — Testing method](../README.md#testing-method).

Run from this sub-example directory:

<pre data-test="exec">
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/linguistics-asia/06-opennlp-icu
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

<pre data-test="exec" data-test-assert-contains="手机壳">
vespa query 'yql=select * from product where id contains "tw-001"' \
            'summary=debug-tokens' 'hits=1'
</pre>

<pre data-test="exec">
docker rm -f vespa
</pre>

## Comparison to other sub-examples

| Sub-example | Cross-variant | Custom Java | Maven build |
|---|---|---|---|
| [02-lucene-per-variant](../02-lucene-per-variant) | no (isolated) | no | yes |
| [03-lucene-icu](../03-lucene-icu) | **yes** (Lucene `icuTransform`) | no | yes |
| [04-jieba](../04-jieba) | no | no (reuses sibling bundle) | no |
| **06 — this one** | **yes** (ICU4J in Java) | **yes** | yes |

Functionally, **03 and 06 give the same cross-variant outcome on this dataset**. Pick 03 if config is enough; pick 06 when you need the Java escape hatch.

## References

- [Vespa `Linguistics` interface](https://github.com/vespa-engine/vespa/blob/master/linguistics/src/main/java/com/yahoo/language/Linguistics.java)
- [Sibling: vespa-chinese-linguistics (same wrapping pattern, Jieba payload)](../../vespa-chinese-linguistics)
- [ICU4J Transliterator IDs](https://unicode-org.github.io/icu/userguide/transforms/general/#transliterator-identifiers)
