<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>


This notebook is work in progress:

```
ERROR: Could not find a version that satisfies the requirement torch==1.* (from versions: 2.0.0, 2.0.1)
ERROR: No matching distribution found for torch==1.*
```

Using torch 2.0 not doable at time of writing:

https://pytorch.org/docs/stable/onnx_supported_aten_ops.html

aten::unflatten.   Not yet supported


# Vespa sample application - Text-Image Search

This sample is an example of a text-to-image search application.
Taking a textual query, such as "two people bicycling",
it will return images containing two people on bikes.
This application is built using [CLIP (Contrastive Language-Image
Pre-Training)](https://github.com/openai/CLIP) which enables "zero-shot prediction".
This means that the system can return sensible results for images it hasn't
seen during training, allowing it to process and index any image.
In this use case, we use the [Flickr8k](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names)
dataset, which was not explicitly used during training of the CLIP model.

This sample application can be used in two different ways.
The first is by using a [Python-based search app](src/python/README.md),
which is suitable for exploration and analysis.
The other is a stand-alone Vespa application, which is more suitable for production (below).
The Python sample app includes a streamlit user interface:

[Animation](https://data.vespa-cloud.com/sample-apps-data/image_demo.gif)

The application takes a textual description and returns the file names of the
images that best match the description. The main difference between this app
and the Python app, is that the transformation from text to a vector
representation has been moved from Python and into Vespa. This includes both
tokenization and transformer model evaluation.

## Quick start
Requirements:
* [Docker](https://www.docker.com/) Desktop installed and running. 6GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* <a href="https://openjdk.org/projects/jdk/17/" data-proofer-ignore>Java 17</a> installed.
* [Apache Maven](https://maven.apache.org/install.html) This sample app uses custom Java components and Maven is used
  to build the application.
* python3.8+ (tested with 3.8)

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

Set up transformer model:
<pre data-test="exec">
$ pip3 install -r src/python/requirements.txt
$ python3 src/python/clip_export.py
</pre>

This extracts the text transformer model from CLIP and puts it into the
`models` directory of the application.

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

Download and extract image data:
<pre>
$ ./src/sh/download_flickr8k.sh
$ export IMG_DIR=data/Flicker8k_Dataset/
</pre>

The full Flickr8k dataset is around 1.1GB, the feed script
uses [PyVespa](https://github.com/vespa-engine/pyvespa/) to feed the image data to the running instance.

Encode images and feed data.
This step take some time as each image is encoded using the CLIP model.
Alternatively use pre-computed embeddings, see next instruction.

<pre>
$ python3 src/python/clip_feed.py
</pre>

Alternatively, instead of computing the embeddings, use the pre-computed embeddings:
<pre data-test="exec">
$ curl -L -o flickr-8k-clip-embeddings.jsonl.zst \
    https://data.vespa-cloud.com/sample-apps-data/flickr-8k-clip-embeddings.jsonl.zst
</pre>

<pre data-test="exec">
$ zstdcat flickr-8k-clip-embeddings.jsonl.zst | vespa feed -
</pre>

Run a query:
<pre data-test="exec" data-test-assert-contains="2337919839_df83827fa0">
$ curl "http://localhost:8080/search/?input=a+child+playing+football&timeout=3s"
</pre>

Shutdown and remove the container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
