<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# SPANN Billion Scale Vector Search

This sample application demonstrates how to represent *SPANN* (Space Partitioned ANN) using Vespa.ai.

The *SPANN* approach for approximate nearest neighbor search is described in
[SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search](https://arxiv.org/abs/2111.08566).

SPANN uses a hybrid combination of graph and inverted index methods for approximate nearest neighbor search.

This sample app demonstrates how the `SPANN` algorithm can be represented using Vespa.
See the [Billion-scale vector search using hybrid HNSW-IF](https://blog.vespa.ai/vespa-hybrid-billion-scale-vector-search/) for details on how `SPANN`
is represented with Vespa.

These reproducing steps, demonstrates the functionality using a smaller subset of the 1B vector dataset, suitable
for reproducing on a laptop.

**Requirements:**

* [Docker](https://www.docker.com/) Desktop installed and running. 6GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* <a href="https://openjdk.org/projects/jdk/17/" data-proofer-ignore>Java 17</a> installed.
* Python3 and numpy to process the vector dataset
* [Apache Maven](https://maven.apache.org/install.html) - this sample app uses custom Java components and Maven is used
  to build the application.

Verify Docker Memory Limits:
<pre>
$ docker info | grep "Total Memory"
or
$ podman info | grep "memTotal"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using docker image use:
<pre data-test="exec">
$ vespa config set target local
</pre>

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/billion-scale-vector-search
</pre>

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
</pre>

Verify that the configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application:
<pre>
$ vespa clone billion-scale-vector-search myapp && cd myapp
</pre>


## Download Vector Data
This sample app uses the Microsoft SPACEV vector dataset from
https://big-ann-benchmarks.com/.

It uses the first 10M vectors of the 100M slice sample.
This sample file is about 1GB (10M vectors):
<pre data-test="exec">
$ curl -L -o spacev10m_base.i8bin \
  https://data.vespa-cloud.com/sample-apps-data/spacev10m_base.i8bin
</pre>

Generate the feed file for the first 10M vectors from the 100M sample.
This step creates two feed files:

* `graph-vectors.jsonl`
* `if-vectors.jsonl`

Install dependencies and create the feed data:
<pre data-test="exec">
$ pip3 install numpy requests tqdm
</pre>
<pre data-test="exec">
$ python3 src/main/python/create-vespa-feed.py spacev10m_base.i8bin
</pre>


## Build and deploy Vespa app
Build the sample app:
<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Deploy the application. This step deploys the application package built in the previous step:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started-java#deploy-sample-applications-java).

Wait for the application endpoint to become available:
<pre data-test="exec">
$ vespa status --wait 300
</pre>

Test basic functionality:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/feed-and-search-test.json
</pre>

The _graph_ vectors must be feed before the _if_ vectors:
<pre data-test="exec">
$ vespa feed graph-vectors.jsonl
$ vespa feed if-vectors.jsonl
</pre>


## Recall Evaluation
Download the query vectors and the ground truth for the 10M first vectors:
<pre data-test="exec">
$ curl -L -o query.i8bin \
  https://github.com/microsoft/SPTAG/raw/main/datasets/SPACEV1B/query.bin
$ curl -L -o spacev10m_gt100.i8bin \
  https://data.vespa-cloud.com/sample-apps-data/spacev10m_gt100.i8bin
</pre>

Note, initially, the routine above used the query file from https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/query.i8bin
but the link no longer works.

Run first 1K queries and evaluate recall@10. A higher number of clusters gives higher recall:
<pre data-test="exec">
$ python3 src/main/python/recall.py --endpoint http://localhost:8080/search/ \
  --query_file query.i8bin \
  --query_gt_file spacev10m_gt100.i8bin  --clusters 12 --queries 1000
</pre>

To evaluate recall using a deployment in Vespa Cloud perf zone, the data plane certificate
and key need to be provided:
<pre>
$ python3 src/main/python/recall.py --endpoint https://app.tenant.aws-us-east-1c.perf.z.vespa-app.cloud/search/ \
  --query_file query.i8bin --query_gt_file GT_10M/msspacev-10M \
  --certificate data-plane-public-cert.pem --key data-plane-private-key.pem
</pre>


## Shutdown and remove the Docker container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
