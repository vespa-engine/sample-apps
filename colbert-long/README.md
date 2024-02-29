
<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Long-Context ColBERT

This semantic search application demonstrates Long-Context ColBERT (multi-token vector representation)
with extended context windows for long-document retrieval. 

The app demonstrates the [colbert-embedder](https://docs.vespa.ai/en/embedding.html#colbert-embedder) and
the tensor expressions for performing two types of extended ColBERT late-interaction for long-context retrieval. 

<p data-test="run-macro init-deploy long-colbert">
Requires at least Vespa 8.311.28
</p>

## To try this application

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started)
through the <code>vespa deploy</code> step, cloning `long-colbert` instead of `album-recommendation`.

Feed documents (this includes embed inference in Vespa):
<pre data-test="exec">
vespa feed ext/sample_feed.jsonl
</pre>

Example query using BM25:
<pre data-test="exec" data-test-assert-contains="id:en:doc::doc-en-16617">
vespa query 'yql=select * from doc where userQuery()'\
 'ranking=bm25' 'hits=1'\
 'query=What is the frequency of Radio KP?'
</pre>

Example query using ColBERT :
<pre data-test="exec" data-test-assert-contains="id:en:doc::doc-en-7562">
vespa query 'yql=select * from doc where userQuery()'\
 'ranking=colbert-max-sim-context-level' 'hits=1' \
 'query=What is the frequency of Radio KP?' \
 'input.query(qt)=embed(colbert, @query)'
</pre>

<pre data-test="exec" data-test-assert-contains="id:en:doc::doc-en-729645">
vespa query 'yql=select * from doc where userQuery()'\
 'ranking=colbert-max-sim-cross-context' 'hits=1'\
 'query=What is the frequency of Radio KP?' \
 'input.query(qt)=embed(colbert, @query)'
</pre>

## Evaluate the effectiveness on long-document retrieval using the MLDR dataset

Install external dependencies:
<pre>
pip3 install datasets langchain
</pre>

Run this script that downloads the MLDR English data split and generates three files; this takes a few minutes (depending on bandwidth).

This simple script writes the feed file to `/tmp/vespa_feed_file_en.json`:
<pre>
python3 scripts/convert.py
</pre>

Index the dataset:
<pre>
vespa feed /tmp/vespa_feed_file_en.json
</pre>

Run the queries (Replace with your endpoint and mTLS cert)
<pre>
python3 evaluate.py --endpoint https://b5af15f0.e2b4d78d.z.vespa-app.cloud/search/ \
  --ranking colbert-max-sim-context-level --dataset ext/test_queries.tsv  --rank_count 10 \
  --key $HOME/.vespa/samples.long-colbert.default/data-plane-private-key.pem \
  --cert$HOME/.vespa/samples.long-colbert.default/data-plane-public-cert.pem
</pre>

Then, evaluate effectiveness by using e.g. `trec_eval`

<pre>
trec_eval -mndcg_cut.10 ext/test_en_qrels.tsv colbert-max-sim-context-level.run 
</pre>

## Terminate
Remove the container after use:
<pre data-test="exec">
$ docker rm -f vespa
</pre>

