<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample application - Text-Image Search

## Short description
Text-Image Search takes a textual description and returns the file names of images that best match the description.
E.g., the query "two people bicycling" will return images with two people on bikes.
See the [blog post](https://blog.vespa.ai/text-image-search/) for more details.


## Features
- CLIP Image embeddings.
- Approximate Nearest Neighbor Search using the `closeness` ranking feature.


## Quick start
This application is built using [CLIP (Contrastive Language-Image Pre-Training)](https://github.com/openai/CLIP),
which enables "zero-shot prediction."
This means that the system can return sensible results for images it hasn't seen during training,
allowing it to process and index any image.

The dataset is  [Flickr8k](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names),
which was explicitly not used during training of the CLIP model.
The transformation from text to a vector representation,
including tokenization and transformer model evaluation,
is handled entirely within Vespa.

Requirements:
* [Docker](https://www.docker.com/) Desktop installed and running. 6GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* [zstd](https://formulae.brew.sh/formula/zstd) installed.
* <a href="https://openjdk.org/projects/jdk/17/" data-proofer-ignore>Java 17</a> installed.
* [Apache Maven](https://maven.apache.org/install.html) This sample app uses custom Java components and Maven is used
  to build the application.

The following instructions sets up the stand-alone Vespa application using the
[Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html).

Validate environment, should be minimum 6G:
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

Checkout this sample app :
<pre data-test="exec">
$ vespa clone text-image-search myapp && cd myapp
</pre>

Compile and run:
<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

<pre data-test="exec">
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 vespaengine/vespa
</pre>

Wait for the configserver to start:
<pre data-test="exec" data-test-assert-contains="is ready">
$ vespa status deploy --wait 300 --color never
</pre>

Deploy the application and wait for services to start:
<pre data-test="exec">
$ vespa deploy --wait 300 --color never
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started-java#deploy-sample-applications-java).

Running [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html)
which runs a set of basic tests to verify that the application is working as expected.
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/image-search-system-test.json
</pre>

This application only returns the names of the images. If you want to view the image returned by Vespa in the next step, you need to download the dataset to access the images. The full Flickr8k dataset is around 1.1GB:
<pre>
$ ./src/sh/download_flickr8k.sh
</pre>

Download and feed the pre-computed embeddings. This is around 40M.
<pre data-test="exec">
$ ./src/sh/download_flickr8k_embeddings.sh
</pre>

<pre data-test="exec">
$ zstdcat embeddings/flickr-8k-clip-embeddings.jsonl.zst | vespa feed -
</pre>

Run a query:
<pre data-test="exec" data-test-assert-contains="2337919839_df83827fa0">
$ vespa query "input=a+child+playing+football&timeout=3s"
</pre>

Shutdown and remove the container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
