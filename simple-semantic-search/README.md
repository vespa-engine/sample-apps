<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - Simple semantic search

A minimal semantic search application: 
- Query and document text is converted to embeddings by the application. 
- Search by embedding and/or text match.

<p data-test="run-macro init-deploy simple-semantic-search">
Requires at least Vespa 8.54.61.
</p>


## To try this

Follow
[vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html)
through the <code>vespa deploy</code> step, cloning simple-semantic-search instead of album-recommendation.

Feed documents:
<pre data-test="exec">
vespa document ext/1.json
vespa document ext/2.json
vespa document ext/3.json
</pre>

Example queries:
<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(space contains many suns)"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(shipping stuff over the sea)"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(exchanging information by sound)"
vespa query "yql=select * from doc where text contains 'boat'"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e) AND text contains 'boat'" "input.query(e)=embed(exchanging information by sound)"
</pre>

Remove the container after use:
<pre data-test="exec">
$ docker rm -f vespa
</pre>


## Ready for production

The [model included in this sample application](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
is suitable for production use and will produce good results in many domains without fine-tuning,
especially when combined with text match features such as bm25.
