
<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Simple hybrid search with ColBERT

This semantic search application uses a single vector embedding model for retrieval and ColBERT (multi-token vector representation)
for re-ranking. It also features reciprocal rank fusion to fuse different rankings.

- Query and document text is converted to embeddings by the application using Vespa's [embedder](https://docs.vespa.ai/en/embedding.html) functionality. 
- Search by embedding or text match, and fuse the rankings each produces using [reciprocal rank fusion](https://docs.vespa.ai/en/phased-ranking.html#cross-hit-normalization-including-reciprocal-rank-fusion). 

<p data-test="run-macro init-deploy colbert">
Requires at least Vespa 8.283.46
</p>

## To try this application

Follow the [vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html)
through the <code>vespa deploy</code> step, cloning `colbert` instead of `album-recommendation`.

Feed documents (this includes embed inference in Vespa):
<pre data-test="exec">
vespa document ext/1.json
vespa document ext/2.json
vespa document ext/3.json
</pre>

Example queries:
<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, q))'\
 'input.query(q)=embed(e5, "query: space contains many suns")' \
 'input.query(qt)=embed(colbert, "space contains many suns")' \
 'query=space contains many suns'
</pre>

<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, q))'\
 'input.query(q)=embed(e5, "query: shipping stuff over the sea")' \
 'input.query(qt)=embed(colbert, "shipping stuff over the sea")' \
 'query=shipping stuff over the sea'
 </pre>

 <pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, q))'\
 'input.query(q)=embed(e5, "query: exchanging information by sound")' \
 'input.query(qt)=embed(colbert, "exchanging information by sound")' \
 'query=exchanging information by sound'
 </pre>


### Terminate container 

Remove the container after use:
<pre data-test="exec">
$ docker rm -f vespa
</pre>

