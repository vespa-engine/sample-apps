<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Multilingual Search with multilingual embeddings

This sample application demonstrates multilingual search
using multilingual embeddings.

Read the [blog post](https://blog.vespa.ai/simplify-search-with-multilingual-embeddings/).

## Quick start

The following is a quick start recipe for getting started with this application.

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
$ vespa clone multilingual-search my-app && cd my-app
</pre>

This sample app embedder configuration in [services.xml](services.xml) points to a quantized model.

Alternatively, [export your own model](https://docs.vespa.ai/en/onnx.html#onnx-export), see also the
export script in [simple-semantic-search](/examples/model-exporting/README.md).

Deploy the application :
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started#deploy-sample-applications).

## Evaluation
The following reproduces the results reported on the MIRACL Swahili(sw) dataset.

Install `trec_eval`:
<pre data-test="exec">
$ git clone --depth 1 --branch v9.0.8 https://github.com/usnistgov/trec_eval && cd trec_eval && make install && cd ..
</pre>

Index the dataset, this also embeds the texts and is compute intensive. On an M1 laptop,
this step takes about 1052 seconds (125 operations/s).

<pre data-test="exec">
$ zstdcat ext/sw-feed.jsonl.zst | vespa feed -
</pre>

The evaluation script queries Vespa (requires pandas and requests libraries):

<pre data-test="exec">
$ pip3 install pandas requests
</pre>

## E5 multilingual embedding model
Using the multilingual embedding model

<pre data-test="exec">
$ python3 ext/evaluate.py --endpoint http://localhost:8080/search/ \
 --query_file ext/topics.miracl-v1.0-sw-dev.tsv \
 --ranking semantic --hits 100 --language sw
 </pre>

Compute NDCG@10 using `trec_eval` with the dev relevance judgments:

<pre data-test="exec" data-test-assert-contains="0.68" >
$ trec_eval -mndcg_cut.10 ext/qrels.miracl-v1.0-sw-dev.tsv semantic.run
</pre>
Which should produce the following:
<pre>
ndcg_cut_10           	all 	0.6848
</pre>

## BM25
Using traditional keyword search with BM25 ranking:

<pre data-test="exec">
$ python3 ext/evaluate.py --endpoint http://localhost:8080/search/ \
 --query_file ext/topics.miracl-v1.0-sw-dev.tsv \
 --ranking bm25 --hits 100 --language sw
 </pre>
Compute NDCG@10 using trec_eval with the same relevance judgments:
<pre data-test="exec" data-test-assert-contains="0.4" >
$ trec_eval -mndcg_cut.10 ext/qrels.miracl-v1.0-sw-dev.tsv bm25.run
</pre>

<pre>
ndcg_cut_10           	all	0.424
</pre>

## Cleanup
Tear down the running container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
