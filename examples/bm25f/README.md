<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# BM25F-inspired cross-field search

Minimal Vespa application implementing BM25F-*inspired* cross-field ranking over `title` and `body` with ranking expressions. "Inspired" rather than BM25F proper, because the document frequency is the term-level estimate for the fieldset — the *sum* of per-field dfs, an upper bound on the document-level df canonical BM25F specifies (docs containing the term in several fields are double-counted).

NOTE: This is only correct for fieldset queries, and the sample query's `text()` retrieves via weakAnd, so with a large corpus and low `targetHits` not every matching document is surfaced for scoring.

The saturation/idf structure — per-term weighted tf summed across fields *before* saturation, one idf per term — is the BM25F part. Built from these ingredients:

- `fieldTermMatch(field, n).occurrences` — term frequency of query term `n` in `field` (per-document; `n` is the global query-term index)
- `fieldLength(field)` and `averageFieldLength(field)` — per-field length normalization
- `queryTermDocumentFrequency(field)` — `tensor(term{})` of the document frequency BM25 would use per query term
- `num_docs_indexed` — corpus size for the IDF

The BM25F score (`rank-profile bm25f`, k1 = 1.2, b = 0.75 for both fields, field weights: title = 2.0, body = 1.0):

```
wtf(t)  = sum_f  weight_f * tf(t, f) / (1 - b_f + b_f * fieldLength(f) / averageFieldLength(f))
df(t)   = queryTermDocumentFrequency(title){term: t}         # term-level (same tensor in every field for fieldset terms)
idf(t)  = log(1 + (num_docs_indexed - df(t) + 0.5) / (df(t) + 0.5))
bm25f   = sum_t  idf(t) * wtf(t) / (k1 + wtf(t))
```

`weighted_tf` is assembled as a literal `tensor(term{})` whose cells call the parameterized function `term_wtf(n)` for term indexes 0..7. Tensor literals are fixed at config time, so this is a cap, not iteration: queries with more than 8 terms have the extra terms silently ignored (raise the cap by adding cells). Unused labels contribute 0 and drop out in the join with `idf`, so any query with up to 8 terms works unchanged — verified with a 3-term query (adding `engine`: doc 1 scores 0.62942 → 1.52058, matching hand computation). Everything downstream (df, idf, score) is generic tensor math.

## Deploy and feed

From the `examples/bm25f` directory:

```bash
vespa config set target local
vespa deploy --wait 300 app
vespa feed dataset/documents.jsonl
```

## Query

```bash
vespa query \
  'yql=select * from bm25f where default contains text("vespa ranking")' \
  'ranking=bm25f'
```

## Document frequency semantics with fieldsets

When a query term searches multiple fields through a fieldset, the query planner gives every term-field the **combined** document frequency estimate — the sum of the per-field dfs — and that is what both `bm25(field)` and `queryTermDocumentFrequency(field)` see. With fieldset queries, `queryTermDocumentFrequency(title)` and `(body)` return identical tensors, and `document_frequency` simply reads one of them — no cross-field combination is needed.

The summed df is an upper bound on the document-level df that canonical BM25F wants (a doc containing the term in both fields is counted twice), and is closer to it than any per-field df. To get true per-field dfs, query each field with separate terms — at the cost of different `term{}` labels per field.
