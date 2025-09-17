<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->
# Deploy your RAG application on your own machine

This shows how you can run your application locally on your own machine.

## Prerequisites

* [Docker](https://www.docker.com/) Desktop installed and running. 10GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro
* Architecture: x86_64 or arm64
* Minimum **8 GB** memory dedicated to Docker (the default is 2 GB on Macs)
* [Vespa-CLI](https://docs.vespa.ai/en/vespa-cli.html) client
  (available for Linux, macOS and Windows, see the link if you don't have Homebrew).


## Optional: Create your application

Do this if you haven't already created your application from the blueprint.
The same Vespa application can be deployed both locally and on Vespa Cloud.

<pre data-test="exec">
$ vespa clone rag-blueprint rag-blueprint && cd rag-blueprint
</pre>

(You can clone into any other name than "rag-blueprint" if you prefer.)

## Configuring for local deployment

<pre data-test="exec">
$ docker run --detach --name rag-blueprint --hostname rag-blueprint \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19112:19112 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
</pre>

<pre data-test="exec">
$ vespa config set target local
</pre>

## Run the application locally

Deploy:

<pre data-test="exec">
$ vespa deploy --wait 300 ./app
</pre>

Feed data:

<pre data-test="exec">
$ vespa feed dataset/docs.jsonl
</pre>

Do a query:

<pre data-test="exec" data-test-assert-contains="yc_b2b_sales_workshop_notes.md">
$ vespa query 'query=yc b2b sales' presentation.summary="no-chunks"
</pre>

## Cleanup
Tear down the running container:
<pre data-test="after">
$ docker rm -f rag-blueprint
</pre>
