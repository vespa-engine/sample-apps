<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa Multi-Vector Indexing with HNSW

This sample application is used to demonstrate multi-vector indexing with Vespa.
Multi-vector indexing was introduced in Vespa 8.144.19. 
Read the [blog post](https://blog.vespa.ai/semantic-search-with-multi-vector-indexing/) announcing multi-vector indexing.

Go to [multi-vector-indexing](https://pyvespa.readthedocs.io/en/latest/examples/multi-vector-indexing.html)
to run this sample application using [pyvespa](https://pyvespa.readthedocs.io/en/latest/index.html).

The app uses a small sample of Wikipedia articles, where each paragraph is embedded in embedding
vector space.

## Quick start

The following is a quick start recipe on how to get started with this application. 

* [Docker](https://www.docker.com/) Desktop installed and running. 4 GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download 
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).

Validate Docker resource settings, should be minimum 4 GB:
<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Verify that configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application:
<pre data-test="exec">
$ vespa clone multi-vector-indexing my-app && cd my-app
</pre>

Download embedding model files, see 
[text embeddings made easy](https://blog.vespa.ai/text-embedding-made-simple/) for details:
<pre data-test="exec"> 
$ mkdir -p model
$ curl -L -o model/tokenizer.json \
  https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/tokenizer.json

$ curl -L -o model/e5-small-v2-int8.onnx \
  https://github.com/vespa-engine/sample-apps/raw/master/simple-semantic-search/model/e5-small-v2-int8.onnx
</pre>

Deploy the application:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started#deploy-sample-applications).


## Indexing sample Wikipedia articles

Index the Wikipedia articles. This embeds all the paragraphs using the native embedding model, which
is computationally expensive for CPU. For production use cases, use [Vespa Cloud with GPU](https://cloud.vespa.ai/en/reference/services#gpu)
instances and [autoscaling](https://cloud.vespa.ai/en/autoscaling) enabled. 

<pre data-test="exec">
$ zstdcat ext/articles.jsonl.zst | vespa feed -
</pre>


## Query and ranking examples
We demonstrate using `vespa cli`, use `-v` to see the curl equivalent using HTTP api.  

### Simple retrieve all articles with undefined ranking
<pre data-test="exec" data-test-assert-contains='"totalCount": 8'>
$ vespa query 'yql=select * from wiki where true' \
  'ranking=unranked'
</pre>

### Traditional keyword search with BM25 ranking on the article level
<pre data-test="exec" data-test-assert-contains='24-hour clock'>
$ vespa query 'yql=select * from wiki where userQuery()' \
  'query=24' \
  'ranking=bm25'
</pre>

Notice the `relevance`, which is assigned by the rank-profile. Also note
that keywords are highlighted in the `paragraphs` field. 

### Semantic vector search on the paragraph level
<pre data-test="exec" data-test-assert-contains='24-hour clock'>
$ vespa query 'yql=select * from wiki where {targetHits:1}nearestNeighbor(paragraph_embeddings,q)' \
  'input.query(q)=embed(what does 24 mean in the context of railways)' \
  'ranking=semantic'
</pre>
The closest (best semantic matching) paragraph has index 4.
```json
"matchfeatures": {
    "closest(paragraph_embeddings)": {"4": 1.0}
}
```
This index corresponds to the following paragraph:
```
"In railway timetables 24:00 means the \"end\" of the day. For example, a train due to arrive at a station during the last minute of a day arrives at 24:00; but trains which depart during the first minute of the day go at 00:00."
```
The [tensor presentation format](search/query-profiles/default.xml) is overridden in
this sample application to shorten down the output. 

### Hybrid search and ranking
Hybrid combining keyword search on the article level with vector search in the paragraph index:

<pre data-test="exec" data-test-assert-contains='24-hour clock'>
$ vespa query 'yql=select * from wiki where userQuery() or ({targetHits:1}nearestNeighbor(paragraph_embeddings,q))' \
  'input.query(q)=embed(what does 24 mean in the context of railways)' \
  'query=what does 24 mean in the context of railways' \
  'ranking=hybrid' \
  'hits=1'
</pre>

This case combines exact search with nearestNeighbor search. The `hybrid` rank-profile 
also calculates several additional features using 
[tensor expressions](https://docs.vespa.ai/en/tensor-user-guide.html):

- `firstPhase` is the score of the first ranking phase, configured in the hybrid
profile as `cos(distance(field, paragraph_embeddings))`.
- `all_paragraph_similarities` returns all the similarity scores for all paragraphs.
- `avg_paragraph_similarity` is the average similarity score across all the paragraphs.
- `max_paragraph_similarity` is the same as `firstPhase`, but computed using a tensor expression.

See the `hybrid` rank-profile in the [schema](schemas/wiki.sd) for details.
The [Vespa Tensor Playground](https://docs.vespa.ai/playground/) is useful to play
with tensor expressions. 

These additional features are 
calculated during [second-phase](https://docs.vespa.ai/en/phased-ranking.html) 
ranking to limit the number of vector computations. 

### Hybrid search and filter

Filtering is also supported, also disable bolding. 

<pre data-test="exec" data-test-assert-contains='24-hour clock'>
$ vespa query 'yql=select * from wiki where url contains "9985" and userQuery() or ({targetHits:1}nearestNeighbor(paragraph_embeddings,q))' \
  'input.query(q)=embed(what does 24 mean in the context of railways)' \
  'query=what does 24 mean in the context of railways' \
  'ranking=hybrid' \
  'bolding=false'
</pre>

## Cleanup
Tear down the running container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
