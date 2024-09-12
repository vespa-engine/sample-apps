
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample application - text search tutorial

This sample application contains the code for the text search tutorial.
Please refer to the [text search tutorial](https://docs.vespa.ai/en/tutorials/text-search.html)
for more information.

See also the [MS Marco Ranking](../msmarco-ranking) sample application for ranking using state-of-the-art retrieval and ranking methods.
There is also a [Ranking with Transformers](../transformers) sample application. 

The following is for deploying the end to end application including a custom front-end.

## Prerequisites

* [Docker](https://www.docker.com/) Desktop installed and running. 10GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64 
* Minimum **10 GB** memory dedicated to Docker (the default is 2 GB on Macs)
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* python 3 
* <a href="https://openjdk.org/projects/jdk/17/" data-proofer-ignore>Java 17</a> installed.
* [Apache Maven](https://maven.apache.org/install.html)

## Installing vespa-cli

This tutorial uses [Vespa-CLI](https://docs.vespa.ai/en/vespa-cli.html),
Vespa CLI is the official command-line client for Vespa.ai.
It is a single binary without any runtime dependencies and is available for Linux, macOS and Windows.

<pre>
$ brew install vespa-cli 
</pre>

<pre data-test="exec">
$ vespa clone text-search text-search && cd text-search
</pre>

<pre data-test="exec">
$ ./bin/convert-msmarco.sh
</pre>

<pre data-test="exec">
$ docker run --detach --name vespa-msmarco --hostname vespa-msmarco \
  --publish 8080:8080 --publish 19112:19112 --publish 19071:19071 \
  vespaengine/vespa
</pre>

<pre data-test="exec">
$ vespa deploy --wait 300 
</pre>

<pre data-test="exec">
$ vespa feed ext/vespa.json
</pre>

<pre data-test="exec" data-test-assert-contains="What Is A  Dad Bod">
$ vespa query 'yql=select title,url,id from msmarco where userQuery()' 'query=what is dad bod' 
</pre>


### Delete container
Remove app and data:
<pre data-test="after">
$ docker rm -f vespa-msmarco
</pre>
