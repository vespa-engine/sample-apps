<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa Vector Streaming Search

This sample application is used to demonstrate vector streaming search with Vespa.
This was introduced in Vespa 8.181.15.
Read the [blog post](https://blog.vespa.ai/announcing-vector-streaming-search/) announcing vector streaming search.
See [Streaming Search](https://docs.vespa.ai/en/streaming-search.html) for more details.

The application uses a small synthetic sample of mail documents for two fictive users.
The subject and content of a mail are combined and embedded into a 384-dimensional embedding space,
using a [Bert embedder](https://docs.vespa.ai/en/reference/embedding-reference.html#bert-embedder).

## Quick start
The following is a quick recipe for getting started with this application.

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

Pull and start the Vespa docker container image:

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
$ vespa clone vector-streaming-search my-app && cd my-app
</pre>

Deploy the application :

<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started#deploy-sample-applications).


## Feeding sample mail documents

During feeding the `subject` and `content` of a mail document are embedded using the Bert embedding model.
This is computationally expensive for CPU.
For production use cases, use [Vespa Cloud with GPU](https://cloud.vespa.ai/en/reference/services#gpu)
instances and [autoscaling](https://cloud.vespa.ai/en/autoscaling) enabled.

<pre data-test="exec">
$ vespa feed ext/docs.json
</pre>

## Query and ranking examples
The following uses [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html) to execute queries.
Use `-v` to see the curl equivalent using HTTP API.

### Exact nearest neighbor search
<pre data-test="exec" data-test-assert-contains='"totalCount": 3'>
$ vespa query 'yql=select * from sources * where {targetHits:10}nearestNeighbor(embedding,qemb)' \
  'input.query(qemb)=embed(events to attend this summer)' \
  'streaming.groupname=1234'
</pre>

This searches all documents for user 1234, and returns the ten best documents
according to the angular distance between the document embedding and the query embedding.

### Exact nearest neighbor search with timestamp filter
<pre data-test="exec" data-test-assert-contains='"totalCount": 2'>
$ vespa query 'yql=select * from sources * where {targetHits:10}nearestNeighbor(embedding,qemb) and timestamp >= 1685577600' \
  'streaming.groupname=1234' \
  'input.query(qemb)=embed(events to attend this summer)'
</pre>

This query only returns documents that are newer than 2023-06-01.

### Exact nearest neighbor search with content filter
<pre data-test="exec" data-test-assert-contains='"totalCount": 1'>
$ vespa query 'yql=select * from sources * where {targetHits:10}nearestNeighbor(embedding,qemb) and content contains "sofa"' \
  'streaming.groupname=5678' \
  'input.query(qemb)=embed(list all order confirmations)'
</pre>

This query only returns documents that match "sofa" in the `content` field.

## Cleanup
Tear down the running container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
