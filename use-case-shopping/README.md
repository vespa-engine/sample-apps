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
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* <a href="https://openjdk.org/projects/jdk/17/" data-proofer-ignore>Java 17</a> installed.
* [Apache Maven](https://maven.apache.org/install.html) This sample app uses custom Java components and Maven is used
  to build the application.
* python3
* zstd: `brew install zstd`

Also read the [Vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html).

Validate environment, should be minimum 4 GB:
<pre>
$ docker info | grep "Total Memory"
# or
$ podman info | grep "memTotal"
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
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
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

Run [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html) -
this runs a set of basic tests to verify that the application is working as expected:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/product-search-test.json
</pre>

Create the data feed for products:
<pre data-test="exec">
$ curl -L -o meta_sports_20k_sample.json.zst https://data.vespa-cloud.com/sample-apps-data/meta_sports_20k_sample.json.zst
$ zstdcat meta_sports_20k_sample.json.zst | ./convert_meta.py > feed_items.json
</pre>

Generate the data feed for reviews:
<pre data-test="exec">
$ curl -L -o reviews_sports_24k_sample.json.zst https://data.vespa-cloud.com/sample-apps-data/reviews_sports_24k_sample.json.zst
$ zstdcat reviews_sports_24k_sample.json.zst | ./convert_reviews.py > feed_reviews.json
</pre>

Generate the data feed for query suggestions:
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

----

### Using Logstash to feed items and reviews

Instead of using `vespa feed`, you can use Logstash to feed items and reviews. This way:
* You can more easily adapt this sample application to your own data. For example, by making Logstash read from different files or other sources, because Logstash is an excellent ETL tool.
* You don't need to convert the reviews to Vespa documents via `./convert_reviews.py`.
* You don't need to convert the items to Vespa documents via `./convert_meta.py` in order to feed them to Vespa. However, this is still needed for suggestions, as `./create_suggestions.py` depends on `feed_items.json`.

You'll need to [install Logstash](https://www.elastic.co/downloads/logstash). Then:

1. Install [Logstash Output Plugin for Vespa](https://github.com/vespa-engine/vespa/tree/master/integration/logstash-plugins/logstash-output-vespa) via:
```
bin/logstash-plugin install logstash-output-vespa_feed
```

2. Change [logstash.conf](logstash.conf) to point to the absolute paths of `meta_sports_20k_sample.json` and `reviews_sports_24k_sample.json`.
   Which still need to be downloaded and uncompressed, as mentioned above:
```
$ curl -L -o meta_sports_20k_sample.json.zst https://data.vespa-cloud.com/sample-apps-data/meta_sports_20k_sample.json.zst
$ unzstd meta_sports_20k_sample.json.zst
$ curl -L -o reviews_sports_24k_sample.json.zst https://data.vespa-cloud.com/sample-apps-data/reviews_sports_24k_sample.json.zst
$ unzstd reviews_sports_24k_sample.json.zst
```

3. Run Logstash with the modified `logstash.conf`:
```
bin/logstash -f $PATH_TO_LOGSTASH_CONF/logstash.conf
```

For more examples of using Logstash with Vespa, check out [this tutorial blog post](https://blog.vespa.ai/logstash-vespa-tutorials/).
