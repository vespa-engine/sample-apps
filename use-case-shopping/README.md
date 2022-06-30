<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - e-commerce

A sample application showcasing a simple e-commerce site built with Vespa.
Refer to [Use Case - shopping](https://docs.vespa.ai/en/use-case-shopping.html).

Included scripts to convert data from Julian McAuley's Amazon product data set
(http://jmcauley.ucsd.edu/data/amazon/links.html) to a Vespa data feed.
This repository contains a small sample of this data from the sports and outdoor category,
but you can download other data from the site above and use the scripts to convert.

### Quick Start 

Requirements:
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
* python3 installed
* zstd: `brew install zstd`

See also [Vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html).

Validate environment, should be minimum 6GB:

<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html).

<pre >
$ brew install vespa-cli
</pre>

Set target env, it's also possible to deploy to [Vespa Cloud](https://cloud.vespa.ai/)
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
$ vespa auth cert
</pre>

See also [Cloud Vespa getting started guide](https://cloud.vespa.ai/en/getting-started). It's possible
to switch between local deployment and cloud deployment by changing the `config target`.

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

<pre data-test="exec">
$ vespa clone use-case-shopping myapp && cd myapp
</pre>

Build the application package
<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Deploy the application package:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

Running [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html)
which runs a set of basic tests to verify that the application is working as expected.

<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/product-search-test.json
</pre>

**Download and create data feed:**

First, create data feed for products:
<pre data-test="exec">
$ curl -L -o meta_sports_20k_sample.json.zst https://data.vespa.oath.cloud/sample-apps-data/meta_sports_20k_sample.json.zst 
$ zstd -d meta_sports_20k_sample.json.zst
$ cat meta_sports_20k_sample.json | ./convert_meta.py > feed_items.json
</pre>

Next, data feed for reviews:
<pre data-test="exec">
$ curl -L -o reviews_sports_24k_sample.json.zst https://data.vespa.oath.cloud/sample-apps-data/reviews_sports_24k_sample.json.zst
$ zstd -d reviews_sports_24k_sample.json.zst 
$ cat reviews_sports_24k_sample.json | ./convert_reviews.py > feed_reviews.json
</pre>

**Feed data:**

Get the [vespa-feed-client](https://docs.vespa.ai/en/vespa-feed-client.html):
<pre data-test="exec">
$ curl -L -o vespa-feed-client-cli.zip \
    https://search.maven.org/remotecontent?filepath=com/yahoo/vespa/vespa-feed-client-cli/7.527.20/vespa-feed-client-cli-7.527.20-zip.zip
$ unzip vespa-feed-client-cli.zip
</pre>

Feed products [vespa-feed-client](https://docs.vespa.ai/en/vespa-feed-client.html):
<pre data-test="exec">
$  ./vespa-feed-client-cli/vespa-feed-client \
    --verbose --file feed_items.json --endpoint http://localhost:8080
</pre>

Feed reviews [vespa-feed-client](https://docs.vespa.ai/en/vespa-feed-client.html):
<pre data-test="exec">
$  ./vespa-feed-client-cli/vespa-feed-client \
    --verbose --file feed_reviews.json --endpoint http://localhost:8080
</pre>

**Test the application:**
<pre data-test="exec" data-test-assert-contains="id:item:item::">
$ vespa query "query=golf"
</pre>

**Browse the site:**
[http://localhost:8080/site](http://localhost:8080/site)

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
