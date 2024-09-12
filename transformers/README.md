<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample application - Transformers

This sample application is a small example of using Transformer-based cross-encoders for ranking
using a small sample from the MS MARCO data set. 

See also the more comprehensive [MS Marco Ranking sample app](../msmarco-ranking/)
which uses multiple Transformer based models for retrieval and ranking. 

This application uses [phased ranking](https://docs.vespa.ai/en/phased-ranking.html), first a set of candidate
documents are retrieved using [WAND](https://docs.vespa.ai/en/using-wand-with-vespa.html). 

The hits retrieved by the WAND operator are ranked using [BM25](https://docs.vespa.ai/en/reference/bm25.html). 
The top-k ranking documents from the first phase
are re-ranked using a cross-encoder Transformer model. 
The cross-encoder re-ranking uses [global phase](https://docs.vespa.ai/en/phased-ranking.html#global-phase), evaluated in the
Vespa stateless container.

## Requirements:

* [Docker](https://www.docker.com/) Desktop installed and running. 4GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64 
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* python3.8+ to export models from Huggingface. 

Validate environment, should be minimum 6G:
<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using container image:
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

Download this sample application:
<pre data-test="exec">
$ vespa clone transformers myapp && cd myapp
</pre>

Install required python packages:
<pre data-test="exec">
$ python3 -m pip install --upgrade pip
$ python3 -m pip install torch transformers onnx onnxruntime
</pre>

For this sample application, we use a [fine-tuned MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) 
model with 6 layers and 22 million parameters.
This step downloads the cross-encoder transformer model, converts it to an ONNX model,
and saves it in the `files` directory:
<pre data-test="exec">
$ ./bin/setup-ranking-model.sh
</pre>

Verify that configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Deploy the app:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300 application
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started#deploy-sample-applications).

Wait for the application endpoint to become available:
<pre data-test="exec">
$ vespa status --wait 300
</pre>

Convert from MS MARCO format to Vespa JSON feed format. 
To use the entire MS MARCO data set, use the download script.
This step creates a `vespa.json` file in the `msmarco` directory:
<pre data-test="exec">
$ ./bin/convert-msmarco.sh
</pre>

Index data:
<pre data-test="exec">
$ vespa feed msmarco/vespa.json
</pre>

Query data.
Note that the embed part is required to convert the query text
to wordpiece representation which is used by the rank-profile:
<pre data-test="exec" data-test-assert-contains="children">
$ vespa query \
 'yql=select title from msmarco where userQuery()' \
 'query=is long term care insurance tax deductible' \
 'ranking=transformer' \
 'input.query(q)=embed(is long term care insurance tax deductible)'
</pre>

This script reads from the MS MARCO queries and issues a Vespa query:
<pre data-test="exec" data-test-assert-contains="children">
$ ./bin/evaluate.py
</pre>

Shutdown and remove the container:
<pre data-test="after">
$ docker rm -f vespa
</pre>


## Bonus 
To export other cross-encoder models, change the code in "src/python/setup-model.py".
However, this sample application uses a Vespa
[WordPiece embedder](https://docs.vespa.ai/en/reference/embedding-reference.html#wordpiece-embedder),
so if the Transformer model requires a different tokenizer, you would have to change the tokenizer. For example
using Vespa [SentencePiece embedder](https://docs.vespa.ai/en/reference/embedding-reference.html#sentencepiece-embedder).
