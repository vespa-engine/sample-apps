<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample applications - e-commerce

A sample application showcasing a simple e-commerce site built with Vespa. Refer to [Use Case - shopping](https://docs.vespa.ai/en/use-case-shopping.html).

Included scripts to convert data from Julian McAuley's Amazon product data set
(http://jmcauley.ucsd.edu/data/amazon/links.html)
to a Vespa data feed. This repository contains a small sample of this data from
the sports and outdoor category, but you can download other data from the site
above and use the scripts to convert.

### How to run

**Validate environment, should be minimum 6G:**
<pre>
$ docker info | grep "Total Memory"
</pre>

**Check-out, compile and run:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/use-case-shopping &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:19071/ApplicationStatus
</pre>

**Deploy the application:**
<pre data-test="exec" data-test-assert-contains="prepared and activated.">
$ curl --header Content-Type:application/zip --data-binary @target/application.zip \
  localhost:19071/application/v2/tenant/default/prepareandactivate
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

Get the feed client:
<pre data-test="exec">
$ curl -L -o vespa-http-client-jar-with-dependencies.jar \
    https://search.maven.org/classic/remotecontent?filepath=com/yahoo/vespa/vespa-http-client/7.391.28/vespa-http-client-7.391.28-jar-with-dependencies.jar
</pre>

Feed products:
<pre data-test="exec">
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --verbose --file feed_items.json --endpoint http://localhost:8080
</pre>

Feed reviews:
<pre data-test="exec">
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --verbose --file feed_reviews.json --endpoint http://localhost:8080
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




