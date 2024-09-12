<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - e-commerce

A sample application showcasing a simple e-commerce site built with Vespa.
See [Use Case - shopping](https://docs.vespa.ai/en/use-case-shopping.html) for features and details:

![Sample app screenshot](https://docs.vespa.ai/assets/img/shopping-1.png)

Also included are scripts to convert data from Julian McAuley's Amazon product data set at
https://cseweb.ucsd.edu/~jmcauley/datasets.html to a Vespa data feed.
This repository contains a small sample of this data from the sports and outdoor category,
but you can download other data from the site above and use the scripts to convert.

Requirements:
* [Docker](https://www.docker.com/) Desktop installed and running. 4 GB available memory for Docker is minimum.
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
* python3
* zstd: `brew install zstd`

See also [Vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html).

Validate environment, should be minimum 4 GB:
<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using Docker image:
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

Verify that configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application:
<pre data-test="exec">
$ vespa clone use-case-shopping myapp && cd myapp
</pre>

Build the application package:
<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Deploy the application package:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started-java#deploy-sample-applications-java).

Run [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html) -
this runs a set of basic tests to verify that the application is working as expected:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/product-search-test.json
</pre>

First, create data feed for products:
<pre data-test="exec">
$ curl -L -o meta_sports_20k_sample.json.zst https://data.vespa-cloud.com/sample-apps-data/meta_sports_20k_sample.json.zst 
$ zstdcat meta_sports_20k_sample.json.zst | ./convert_meta.py > feed_items.json
</pre>

Next, data feed for reviews:
<pre data-test="exec">
$ curl -L -o reviews_sports_24k_sample.json.zst https://data.vespa-cloud.com/sample-apps-data/reviews_sports_24k_sample.json.zst
$ zstdcat reviews_sports_24k_sample.json.zst | ./convert_reviews.py > feed_reviews.json
</pre>

Next, data feed for query suggestions:
<pre data-test="exec">
$ pip3 install spacy mmh3
$ python3 -m spacy download en_core_web_sm 
$ ./create_suggestions.py feed_items.json > feed_suggestions.json
</pre>

Feed products data:
<pre data-test="exec">
$ vespa feed feed_items.json
</pre>

Feed reviews data:
<pre data-test="exec">
$ vespa feed feed_reviews.json
</pre>

Feed query suggestions data:
<pre data-test="exec">
$ vespa feed feed_suggestions.json
</pre>

Test the application:
<pre data-test="exec" data-test-assert-contains="id:item:item::">
$ vespa query "query=golf"
</pre>

Browse the site:
[http://localhost:8080/site](http://localhost:8080/site)

Shutdown and remove the container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
