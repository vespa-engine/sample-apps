<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Customizing Frozen Data Embeddings in Vespa

This sample application is used to demonstrate how to adapt frozen embeddings from foundational
embedding models.
Frozen data embeddings from Foundational models are an emerging industry practice for reducing the complexity of maintaining and versioning embeddings. The frozen data embeddings are re-used for various tasks, such as classification, search, or recommendations.

Read the [blog post]([blog post](https://blog.vespa.ai/tailoring-frozen-embeddings-with-vespa/)).


## Quick start
The following is a quick start recipe on how to get started with this application:

* [Docker](https://www.docker.com/) Desktop installed and running. 4 GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).

Validate Docker resource settings, should be minimum 4 GB:
<pre>
$ docker info | grep "Total Memory"
or
$ podman info | grep "memTotal"
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
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
</pre>

Verify that configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application:
<pre data-test="exec">
$ vespa clone custom-embeddings my-app && cd my-app
</pre>

Download a frozen embedding model file, see
[text embeddings made easy](https://blog.vespa.ai/text-embedding-made-simple/) for details:
<pre data-test="exec">
$ mkdir -p models
$ curl -L -o models/tokenizer.json \
  https://raw.githubusercontent.com/vespa-engine/sample-apps/master/examples/model-exporting/model/tokenizer.json

$ curl -L -o models/frozen.onnx \
  https://github.com/vespa-engine/sample-apps/raw/master/examples/model-exporting/model/e5-small-v2-int8.onnx

$ cp models/frozen.onnx models/tuned.onnx
</pre>

In this case, we re-use the frozen model as the tuned model to demonstrate functionality.

Deploy the application :
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started#deploy-sample-applications).


## Indexing sample documents
<pre data-test="exec">
vespa document ext/1.json
vespa document ext/2.json
vespa document ext/3.json
</pre>

## Query and ranking examples

We demonstrate using `vespa cli`, use `-v` to see the curl equivalent using HTTP api.

### Simple retrieve all documents with undefined ranking:
<pre data-test="exec" data-test-assert-contains='"totalCount": 3'>
vespa query 'yql=select * from doc where true' \
'ranking=unranked'
</pre>
Notice the `relevance`, which is assigned by the rank-profile.

### Using the frozen query tower
<pre data-test="exec" data-test-assert-contains='"totalCount": 3'>
vespa query 'yql=select * from doc where {targetHits:10}nearestNeighbor(embedding, q)' \
'input.query(q)=embed(frozen, "space contains many suns")'
</pre>

### Using the tuned query tower
<pre data-test="exec" data-test-assert-contains='"totalCount": 3'>
vespa query 'yql=select * from doc where {targetHits:10}nearestNeighbor(embedding, q)' \
'input.query(q)=embed(tuned, "space contains many suns")'
</pre>
In this case, the tuned model is equivelent to the frozen query tower that was used for document embeddings.

### Using the simple weight transformation query tower
<pre data-test="exec" data-test-assert-contains='"totalCount": 3'>
vespa query 'yql=select * from doc where {targetHits:10}nearestNeighbor(embedding, q)' \
'input.query(q)=embed(tuned, "space contains many suns")' \
'ranking=simple-similarity'
</pre>
This invokes the `simple-similarity` ranking model, which performs the query transformation
to the tuned embedding.

### Using the Deep Neural Network similarity
<pre data-test="exec" data-test-assert-contains='"totalCount": 3'>
vespa query 'yql=select * from doc where {targetHits:10}nearestNeighbor(embedding, q)' \
'input.query(q)=embed(tuned, "space contains many suns")' \
'ranking=custom-similarity'
</pre>

Note that this just demonstrates the functionality, the custom similarity model is
initialized from random weights.

### Dump all embeddings
This is useful for training routines, getting the frozen document embeddings out of Vespa:
<pre>
vespa visit --field-set "[all]" > ../vector-data.jsonl
</pre>

### Get a specific document and it's embedding(s):
<pre>
curl "http://localhost:8080/document/v1/doc/doc/docid/1?fieldSet=\[all\]"
</pre>


## Cleanup
Tear down the running container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
