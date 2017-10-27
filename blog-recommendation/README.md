<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - blog recommendation tutorial

This sample application contains the code for the blog recommendation tutorial.

Please refer to
[Vespa tutorial pt.2 - Blog recommendation](http://docs.vespa.ai/documentation/tutorials/blog-recommendation.html)
for more information.

### Executable example
**Check-out, compile and run:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/blog-recommendation &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/blog-recommendation/target/application.zip &amp;&amp; \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
**Test the application:**
<pre data-test="exec" data-test-assert-contains='"coverage":100,"documents":0'>
$ curl 'http://localhost:8080/search/?searchChain=blog&amp;user_item_cf=%7B%7Buser_item_cf%3A0%7D%3A0.1%2C%7Buser_item_cf%3A1%7D%3A0.1%2C%7Buser_item_cf%3A2%7D%3A0.1%2C%7Buser_item_cf%3A3%7D%3A0.1%2C%7Buser_item_cf%3A4%7D%3A0.1%2C%7Buser_item_cf%3A5%7D%3A0.1%2C%7Buser_item_cf%3A6%7D%3A0.1%2C%7Buser_item_cf%3A7%7D%3A0.1%2C%7Buser_item_cf%3A8%7D%3A0.1%2C%7Buser_item_cf%3A9%7D%3A0.1%7D'
</pre>
**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
