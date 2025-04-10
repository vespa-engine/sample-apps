
<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Simple hybrid search with SPLADE

This semantic search application combines BM25 with SPLADE
for re-ranking. The sample app demonstrates the [splade-embedder](https://docs.vespa.ai/en/embedding.html#splade-embedder).

This sample application demonstrates using an apache 2.0 licenced splade model checkpoint 
[prithivida/Splade_PP_en_v1](https://huggingface.co/prithivida/Splade_PP_en_v1). 

The original SPLADE repo and model checkpoints have restrictive licenses:

- HF model checkpoint [naver/splade-v3](https://huggingface.co/naver/splade-v3)
- GitHub naver splade repo [naver/splade](https://github.com/naver/splade)

There is a growing number of independent 
open-source sparse encoder checkpoints that are compatible with the Vespa splade embedder implementation:

- [opensearch-project/opensearch-neural-sparse-encoding-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v1) 
- [Neural-Cherche is a library designed to fine-tune neural search models such as Splade](https://github.com/raphaelsty/neural-cherche)

See [exporting fill-mask language models to onnx format](#exporting-fill-mask-models-to-onnx).


<p data-test="run-macro init-deploy splade">
Requires at least Vespa 8.320.68
</p>

## To try this application

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started)
through the <code>vespa deploy</code> step, cloning `splade` instead of `album-recommendation`.


## Indexing sample documents

<pre data-test="exec">
$ vespa feed ext/*.json
</pre>


## Query examples
We demonstrate queries using the `vespa-cli` tool,
 please use `-v` to see the curl equivalent using the HTTP query API.  


<pre data-test="exec" data-test-assert-contains='id:doc:doc::3'>
$ vespa query 'query=stars' \
 'input.query(q)=embed(splade,@query)' \
 'presentation.format.tensors=short-value'
</pre>

Which will produce the following hit output

```json
{
    "id": "id:doc:doc::3",
    "relevance": 32.3258056640625,
    "source": "text",
    "fields": {
        "matchfeatures": {
            "bm25(chunk)": 1.041708310095213,
            "bm25(title)": 0.9808292530117263,
            "query(q)": {
                "who": 1.1171875,
                "star": 2.828125,
                "stars": 2.875,
                "sky": 0.9375,
                "planet": 0.828125
            },
            "chunk_token_scores": {
                "star": 7.291259765625,
                "stars": 7.771484375,
                "planet": 0.7310791015625
            },
            "title_token_scores": {
                "star": 8.086669921875,
                "stars": 8.4453125
            }
        },
        "sddocname": "doc",
        "documentid": "id:doc:doc::3",
        "splade_chunk_embedding": {
            "the": 0.84375,
            "with": 1.3671875,
            "star": 2.578125,
            "stars": 2.703125,
            "filled": 2.171875,
            "planet": 0.8828125,
            "universe": 1.4296875,
            "fill": 2.03125,
            "filling": 1.5546875,
            "galaxy": 2.765625,
            "galaxies": 1.7265625
        },
        "splade_title_embedding": {
            "about": 1.984375,
            "star": 2.859375,
            "stars": 2.9375,
            "documents": 1.8671875,
            "starred": 0.81640625,
            "document": 2.671875,
            "concerning": 0.8671875
        },
        "title": "A document about stars",
        "chunk": "The galaxy is filled with stars"
    }
}
```

The `rank-profile` used here is `default`, specified in the [schemas/doc.sd](app/schemas/doc.sd) file. 

It includes a [match-features](https://docs.vespa.ai/en/reference/schema-reference.html#match-features) configuration
specifying tensor and rank-features we want to return with each hit. We have:

- `bm25(title)` the bm25 score of the query, title pair
- `bm25(chunk)` the bm25 score of the query, chunk pair
- `query(q)` - the splade query tensor produced by the embedder with all the tokens and their corresponding weight
- `splade_chunk_embedding` - the mapped tensor produced by the embedder at indexing time (chunk)
- `splade_title_embedding` - the mapped tensor produced by the embedder at indexing time (title)
- `chunk_token_scores` - the non-zero overlap between the mapped query tensor and the mapped chunk tensor 
- `title_token_scores` - same as above, but for the title

The last two outputs allow us to highlight the terms of the source text for explainability. 

Note that this application sets a high `term-score-threshold` to reduce the output verbosity. This
setting controls which tokens are retained and used in the dot product calculation(s). 

A higher threshold increases sparseness and reduces complexity and accuracy. 

<pre data-test="exec" data-test-assert-contains='id:doc:doc::1'>
$ vespa query 'query=boats' \
 'input.query(q)=embed(splade,@query)' \
 'presentation.format.tensors=short-value'
</pre>

<pre data-test="exec" data-test-assert-contains='id:doc:doc::2'>
$ vespa query 'query=humans talk a lot' \
 'input.query(q)=embed(splade,@query)' \
 'presentation.format.tensors=short-value'
</pre>

### Retrieval versus ranking

Note that in this sample application, Vespa is not using the expanded sparse learned weights for retrieval (matching).

It's used in a [phased ranking](https://docs.vespa.ai/en/phased-ranking.html) pipeline where we *retrieve* efficiently using 
Vespa's [weakAnd algorithm](https://docs.vespa.ai/en/using-wand-with-vespa.html) with [BM25](https://docs.vespa.ai/en/reference/bm25.html). 

This phased ranking pipeline considerably speeds up retrieval compared to using the lexical expansion. 
It's also possible to retrieve/query using the `wand` vespa query operator. See an example in 
the documentation about using the [wand](https://docs.vespa.ai/en/using-wand-with-vespa.html#wand).

We can also brute-force score and rank all documents that match a filter, this can also be
accelerated by [using multiple search threads per query](https://search.vespa.ai/search?query=using%20multiple%20threads%20per%20search). 

<pre>
vespa query 'yql=select * from doc where true' \
 'input.query(q)=embed(night sky of stars)' \
 'presentation.format.tensors=short-value'
</pre>
 
 For longer contexts using array inputs, see the tensor playground example for scoring options.

 [playground splade tensors in ranking](https://docs.vespa.ai/playground/#N4KABGBEBmkFxgNrgmUrWQPYAd5QGNIAaFDSPBdDTAF30gEJEA1AUwGccBDMAGywBzLAF0AFAAtatHBzgB6eQBNutbgDoAbpx7qsqieoICArkvncOHNrQ7ztXbgFoBwpwCM+3AgGsnAZgAGQPUcADtBAEoAHTCAYjAAYSwAWxwTWgBLCLAAd0zaCTAAZQAFABkAQQARAFEwWjYwjiwAJw4wbLB2RzBW7jCfbMFY2IAVCUyOpTYUrGbafsaOkw5hhqaW9rAcVqwlEwI2JTB3AE8GiTYkLi8Zp1n3Y5nW8Vvue8fntlbIzrC+gMhhF1CUcHdrlNTpZjmB5mBNANMnwvA1+s1oG0Uj8wHMZnwOvlCmBrABHExNI78JqCYl8TIpAqqTLzDjjSYdVpscmZLkdAgSEyDda0LA7PYHKnYpERaAmPh9Ni7ThNNRZVlw6D8eaCHGNAAetlBAFUwvSfNc1hE+GwnNoCKLWmAVGp3DCOMRujpeNwUVhcpylXzVcNYoVrgKhT4OlgteG4a1MoJsr7tTkDbQ8gUJFgMiTwQUsumrmnBBtDf9RWAAFTV6w8JZsWuKxaZNiIzzXIUFDjqUYAiZQ30tMC+gQBkkENoikshrml8uZ7IcRofTVgABWq0z8cjg2p2j4oMHHXB3DOgj2QpObH13DSNuds1Zi1UnFOWGJ3Bwu30AvfmKtLEHBTomOQCDkShYAQJjYmEthZsSxRnjMirKtY8HMqyoL9gAYm0YC6mEPyptkiz7Ic6oAlWjTNG0HRdD0PBwLEThIGMmwEWREqUSyAKCCYmQzOIUgyHIihQQQvYOLo3CZPITTyLRWxOKsPxOAJQlsOoUgpHwMRhGxiAcXRTq7AyBSZNoYBymEDp8RwInSLICjKNB0neuockKWE8hctAPyUmwflAsMDz6uhazYbpfBxOZjJZNoTi2fZrIGUZJlbGAYTzE48WWdZKVUY5kjOeJblSVonneYp-mBXZwXosCgjhZFDk6bQelxDlhn5YltpFQ56VIAAQq4CDJGkGTrESRTKfR-xeo4Tlia5nhCFVjhefJU5TUWLWzU483tPIBn9rEADq1wSNw1m0LkYopN+OCwsdcj9mxAAG5I-GcYikrQkSfXkkwCp0HS8AAjCcT0-q9nFmTxRwnOcYBfEoSjrLwP2tBcs1gJ96NiO8MyetEkBhEmUgkj4FyxiSajtOTQOsQTqitu4GRsGIe7RkDnq5KDRRDmAABMMPPfDpnihRyOnBc6OYzkAyjq0-QXNk6SZqo-wzPqIoMmwLFhLEn1m4goyQBM1yCL63D6hrHTQMiNonPjK7cEzJCW8kQq0DaVgM570yfpcNt2w7ADkHRNEoAcdJTgjUxwtN9pAsQiGbn39tbDQPRspkdDd1nzJaN1cicmNwVFAKfaKFphMDguZGDXLoaqHTxt2P0NFgDc2XsKRh2iAwcIB2JOpo0HcJzXi4+okBkKgAC+S-L6QGDUOQuAMGwJBLxAFD4FvNAUwwON-QDkT7zQmB7wgkDHQAPNAAiqAAfGI9dNMAy+RHAwBYi32ASA4BsYECQ3UAAFn8AAdgAKzECAaAlBIDE5SAQKLdQsCACcAA2AAHAgpBAJUFkIwB7VoED1A4MCHg0WiDkHkPIZQuQYt1Ci3wUQxhpDmFkJTmcTB6g4E8L4eQ9BtAEAhAIZDPBUC6EMJIWI-hQxODUPgZDeB9D4FMOURgZei9b6r30RvVAJ9D47wfnvUx5BKBoAPpgMIDB2aJk5o0HmgpBgcGvjY0+98oDP1fvoWgn9ea-09N-MIv9-5oFiMAMAvM4CBGIJEuA4Zl5JPUAQmBxC4kJM8T4JJKS+5NDgLNDJkD-B4NgZDbhJD4mJOSakyhGSsHwNgTIxReTGnFIbnAVhrTsGBH8JDLpUT8lRiKak52KJjiDMhjUup3SClTJKWEOAZ5iK0AydIghotOk8IaSsppay4Ddm0O0NgFToGcMIbk8ZPTpku0GcM0ZhyJmDFWX0mZ9IIjXPgfAqBdz3mPNObbLwDtBmwLwVosZRzJknL6eC+2bYODXNgaLGF2j6kfMKZDXppSnphDONcwIgLgU4sSfi1JEjrlAqWQ8lZ1LTktLgJA0WkNAg4JEZSplBL1kDLZeoDRsKQV8ppbBH4uY0WZNwRS5ZkzmV9IEdcyG-h9lvN5Yq-lcBkWQqFfAkZcLcVsp1bHeOdKeUKs+Uq0pUFtlCtuQy+FNqdVcH+UasV2rUlTj9haoV8jsXWrxTqu48zA0MP7AYhxxiV6+PMVASxhAb6nzsQmw+9AH5xASMUUC6xVgigRmjCKfIa5shNmEK6YBi7XHumKHAWArCZHcMiSyAECIgWnMWa4klJXwRsgReMF9jas2NFaMsABZe2SQClgHKO2NgCpc1tG0mAE8Owfjj07iWe1v4DiZhek6Xmo4wgnHjL2uCmZO3zjUtu64T19YpFgrig8i7Jwrr7IZMAY71hTv1GutZc6F1LtAqu3OHBn303jA+3u-cbSHnfXyUcBA9iB15r2QxNBY0QHXkvBN2A7GQGsQ4o+VAHGHycQ-XdSN6C+PIP4yAF9-qA2rC45tXMPFRm8Zh-Ra940kaTURlNtjj7kagJRqAud7Uy0lJmQCo4UTD3Q8JvxDAIMpDENR2WtAIlrOvjGvjeGBOEeI7fUj9iQFnwftUaCfar2gfBsPGDUGSzHvg2+69nAVP0YYBXQ43N1Oac-Huyi-LIiegfZ6Xm+mjGGc3sZhgRA6NQDTWJx+DBs1gEqHwRorQwjMmste9YOcSyYj9PkSCz4FiNhWOOpz07ImIffN4VDd7cUcB4yvOLZiEtWO8yl0TlmJOQEqIIS8bBbaNDhBcpTBTavrAff1w+DG-NHCCzIGjEX7ZRYKTFrD3WID4cE6Z1Ng3gFWagDZmCl6muOfjOp9c0GGuAfcwqTzHXkvLbU7BMQq3uZadk1t-UO2ozhdg00PbvGodgBECAZeQA)


### Exporting fill-mask models to onnx

To export a model trained with `fill-mask` (compatible with the `splade-embedder`):

<pre>
$ pip3 install optimum onnx 
</pre>

Export the model using the `optimum-cli` with task `fill-mask`:

<pre>
$ optimum-cli export onnx --task fill-mask --model the-splade-model-id output
</pre>

Remove the exported model files that are not needed by Vespa:

<pre>
$ find models/ -type f ! -name 'model.onnx' ! -name 'tokenizer.json' | xargs rm
</pre>


#### Terminate container 

This is only relevant when running this sample application locally. Remove the container after use:
<pre data-test="after">
$ docker rm -f vespa
</pre>
