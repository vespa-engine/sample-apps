<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# SPANN Billion Scale Vector Search 

This sample application demonstrates how to represent *SPANN* (Space Partitioned ANN) using Vespa.ai. 

The *SPANN* is described in
[SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search](https://arxiv.org/abs/2111.08566). 

SPANN uses a hybrid combination of graph and inverted index methods for approximate nearest neighbor search. 

This sample app demonstrates how SPANN can be represented using Vespa:

* Randomly select 20% of the vector data to represent centroids. As demonstrated in the paper, random selection works
reasonably well but using offline clustering can improve recall and performance. 
The huge benefit of random selection is that it just need one pass through the dataset
and avoids resource intensive offline clustering. 
* Index the vectors which are assigned centroids using Vespa's HNSW indexing support in a dedicated 
content cluster. This content cluster is scaled using instance types with high memory since HNSW indexing
require the vector data to be in-memory for best indexing and search performance.  
* 
  
Requirements:

* [Docker](https://www.docker.com/) Desktop installed and running. 6GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [Github releases](https://github.com/vespa-engine/vespa/releases).
* [Java 11](https://openjdk.java.net/projects/jdk/11/) installed.
* Python3 and numpy to process the vector dataset 
* [Apache Maven](https://maven.apache.org/install.html) This sample app uses custom Java components and Maven is used
  to build the application. 

Verify Docker Memory Limits:

<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html).

<pre >
$ brew install vespa-cli
</pre>

Set target env, it's also possible to deploy this app to [Vespa Cloud](https://cloud.vespa.ai/)
using target cloud.

For local deployment using docker image use

<pre data-test="exec">
$ vespa config set target local
</pre>

For cloud deployment using [Vespa Cloud](https://cloud.vespa.ai/) use
<pre>
$ vespa config set target cloud
$ vespa config set application tenant-name.myapp.default
$ vespa auth login 
$ vespa cert
</pre>

See also [Cloud Vespa getting started guide](https://cloud.vespa.ai/en/getting-started). It's possible
to switch between local deployment and cloud deployment by changing the `config target`.

<pre style="display:none" data-test="exec">
$ pwd && ls
$ cd sample-apps/billion-scale-vector-search
</pre>

Pull and start the vespa docker container image:

<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Verify that configuration service (deploy api) is ready

<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application

<pre>
$ vespa clone billion-scale-vector-search myapp && cd myapp
</pre>


## Download Vector Data

This sample app uses the Microsoft SPACEV vector dataset from 
[https://big-ann-benchmarks.com/](https://big-ann-benchmarks.com/]) 

This quick start uses the first 10M vectors of the 100M slice sample. 

This sample file is about 1GB (10M vectors)

<pre data-test="exec">
$ curl -L -o spacev10m_base.i8bin \
  https://data.vespa.oath.cloud/sample-apps-data/spacev10m_base.i8bin
</pre>

Generate the feed file for the first 10M vectors from the 100M sample. 
This step creates two feed files:

* `graph-vectors.jsonl`
* `if-vectors.jsonl`

<pre data-test="exec">
$ python3 src/main/python/create-vespa-feed.py spacev10m_base.i8bin
</pre>

## Build and deploy Vespa app 

Build the sample app 

<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Deploy the application. This step deploys the application package built in the previous step:

<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

Wait for the application endpoint to become available

<pre data-test="exec">
$ vespa status --wait 300
</pre>

Feed the vector using the [vespa-feed-client](https://docs.vespa.ai/en/vespa-feed-client.html):

<pre data-test="exec">
$ curl -L -o vespa-feed-client-cli.zip \
    https://search.maven.org/remotecontent?filepath=com/yahoo/vespa/vespa-feed-client-cli/7.588.57/vespa-feed-client-cli-7.588.57-zip.zip
$ unzip -o vespa-feed-client-cli.zip
</pre>

The graph vectors must be feed before the if vectors:

<pre data-test="exec">
$ ./vespa-feed-client-cli/vespa-feed-client \
  --file graph-vectors.jsonl --endpoint http://localhost:8080/
$ ./vespa-feed-client-cli/vespa-feed-client \
  --file if-vectors.jsonl --endpoint http://localhost:8080/
</pre>

## Recall Evaluation
Download the query vectors and the ground truth for the 10M first vectors 
<pre data-test="exec">
$ curl -L -o query.i8bin \
https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/query.i8bin
$ curl -L -o spacev10m_gt100.i8bin \
  https://data.vespa.oath.cloud/sample-apps-data/spacev10m_gt100.i8bin
</pre>

Run first 1K queries and evaluate recall@10. Higher number of clusters gives higher recall 

<pre data-test="exec">
$ python3 src/main/python/recall.py --endpoint http://localhost:8080/search/ \
--query_file query.i8bin \
--query_gt_file spacev10m_gt100.i8bin  --clusters 12 --queries 1000
</pre>

To evaluate recall using a deployment in Vespa Cloud perf zone the data plane certificate
and key needs to be provided:

<pre>
python3 src/main/python/recall.py --endpoint https://app.tenant.aws-us-east-1c.perf.z.vespa-app.cloud/search/ \
 --query_file query.i8bin --query_gt_file GT_10M/msspacev-10M \ 
 --certificate data-plane-public-cert.pem --key data-plane-private-key.pem
</pre>

<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/feed-and-search-test.json
</pre>

## Shutdown and remove the Docker container:

<pre data-test="after">
$ docker rm -f vespa
</pre>