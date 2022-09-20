<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - Simple semantic search

A minimal semantic search application: 
- Query and document text is converted to embeddings by the application. 
- Search by embedding and/or text match.

Requested at least Vespa 8.54.61.

## To try this

Follow
[vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html)
through the <code>vespa deploy</code> step, cloning simple-semantic-search instead of album-recommendation.

Feed documents:
<pre>
vespa document ext/1.json
vespa document ext/2.json
vespa document ext/3.json
</pre>

Example queries:
<pre>
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(space contains many suns)"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(shipping stuff over the sea)"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(exchanging information by sound)"
vespa query "yql=select * from doc where text contains 'boat'"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e) AND text contains 'boat'" "input.query(e)=embed(exchanging information by sound)"
</pre>
