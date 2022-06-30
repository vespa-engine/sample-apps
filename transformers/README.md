<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample application - Transformers

This sample application is a small example of using Transformers for ranking
using a small sample from the MS MARCO data set. 
See also the more comprehensive [MS Marco Ranking sample app](../msmarco-ranking/)
which uses multiple Transformer based models for retrieval and ranking. 

This application uses [phased ranking](https://docs.vespa.ai/en/phased-ranking.html), first a set of candidate
documents are retrieved using [WAND](https://docs.vespa.ai/en/using-wand-with-vespa.html) and the first phase ranking
is using [BM25](https://docs.vespa.ai/en/reference/bm25.html). The top-k ranking documents from the first phase
is re-ranked using the Transformer model. 

## Requirements:

* [Docker](https://www.docker.com/) Desktop installed and running. 6GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [Github releases](https://github.com/vespa-engine/vespa/releases).
* [Java 11](https://openjdk.java.net/projects/jdk/11/) installed.
* [Apache Maven](https://maven.apache.org/install.html) This sample app uses custom Java components and Maven is used
  to build the application.
* python3.8+ (tested with 3.8)

**Validate environment, should be minimum 6G:**

Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
for details and troubleshooting:

<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html).

<pre >
$ brew install vespa-cli
</pre>

Set target env, it's also possible to deploy to [Vespa Cloud](https://cloud.vespa.ai/)
using target cloud.

For local deployment using container image use

<pre data-test="exec">
$ vespa config set target local
</pre>

For cloud deployment using [Vespa Cloud](https://cloud.vespa.ai/) use

<pre>
$ vespa config set target cloud
$ vespa config set application tenant-name.myapp.default
$ vespa auth login 
$ vespa auth cert
</pre>

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Download this sample application

<pre data-test="exec">
$ vespa clone transformers myapp && cd myapp
</pre>

Install required python packages:

<pre data-test="exec">
$ python3 -m pip install --upgrade pip
$ python3 -m pip install torch transformers onnx onnxruntime
</pre>

**Setup ranking models**

This downloads the transformer model, converts it to an ONNX model and puts it
in the `files` directory. 

For this sample application, we use a [fine-tuned MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) 
model with 6 layers and 22 million parameters. However other
[Transformers models](https://huggingface.co/transformers/index.html) can be
used. 

To export other models, for instance DistilBERT or ALBERT, change the
code in "src/python/setup-model.py". However, this sample application
uses a Vespa [WordPiece embedder](https://docs.vespa.ai/en/embedding.html), so if the Transformer model requires a
different tokenizer, you would have to add that yourself.

<pre data-test="exec">
$ ./bin/setup-ranking-model.sh
</pre>

**Build the application package:**
<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Verify that configuration service (deploy api) is ready

<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>


Deploy the application 

<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

Wait for the application endpoint to become available

<pre data-test="exec">
$ vespa status --wait 300
</pre>


**Create data feed:**

Convert from MS MARCO format to Vespa JSON feed format. 
To use the entire MS MARCO data set, use the download script.

<pre data-test="exec">
$ ./bin/convert-msmarco.sh
</pre>

**Feed data:**

<pre data-test="exec">
$ curl -L -o vespa-feed-client-cli.zip \
    https://search.maven.org/remotecontent?filepath=com/yahoo/vespa/vespa-feed-client-cli/7.527.20/vespa-feed-client-cli-7.527.20-zip.zip
$ unzip vespa-feed-client-cli.zip
$ ./vespa-feed-client-cli/vespa-feed-client \
    --verbose --file msmarco/vespa.json --endpoint http://localhost:8080
</pre>


**Test the application:**

Running [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html)
which runs a set of basic tests to verify that the application is working as expected.

<pre data-test="exec" data-test-assert-contains="Success">
vespa test src/test/application/tests/system-test/document-ranking-test.json
</pre>

This script reads from the MS MARCO queries and issues a Vespa query:

<pre data-test="exec" data-test-assert-contains="children">
$ ./src/python/evaluate.py
</pre>

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
