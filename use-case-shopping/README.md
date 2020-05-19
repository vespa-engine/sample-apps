<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample applications - e-commerce

A sample application showcasing a simple e-commerce site built with Vespa. Refer to [Use Case - shopping](https://docs.vespa.ai/documentation/use-case-shopping.html).

Included scripts to convert data from Julian McAuley's Amazon product data set
(http://jmcauley.ucsd.edu/data/amazon/links.html)
to a Vespa data feed. This repository contains a small sample of this data from
the sports and outdoor category, but you can download other data from the site
above and use the scripts to convert.

### How to run

**Check-out, compile and run:**

<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/use-case-shopping &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>

**Wait for the configserver to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

**Deploy the application:**

<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/use-case-shopping/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>

**Wait for the application to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Create data feed:**

First, create data feed for products:

<pre data-test="exec">
$ gunzip -c meta_sports_20k_sample.json.gz | ./convert_meta.py > feed_items.json
</pre>

Next, data feed for reviews:

<pre data-test="exec">
$ gunzip -c reviews_sports_24k_sample.json.gz | ./convert_reviews.py > feed_reviews.json
</pre>

**Feed data:**

Feed products:

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /vespa-sample-apps/use-case-shopping/feed_items.json --host localhost --port 8080'
</pre>

Feed reviews:

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /vespa-sample-apps/use-case-shopping/feed_reviews.json --host localhost --port 8080'
</pre>

**Test the application:**

<pre data-test="exec" data-test-assert-contains="id:item:item::">
$ curl -s http://localhost:8080/search/?query=default:golf
</pre>

**Browse the site:**

[http://localhost:8080/site](http://localhost:8080/site)

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>




