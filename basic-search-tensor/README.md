<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - Basic search with tensors

A basic Vespa application, which supports feeding and running simple queries
using tensors.

Also includes the rank expression playground used to visualize rank operations. This sample application is based on [basic-search-java](https://github.com/vespa-engine/sample-apps/tree/master/basic-search-java) which requires building the application using maven before deploying. See [Developing application](http://docs.vespa.ai/documentation/jdisc/developing-applications.html). Once deployed you can view the tensor playground:

    http://<host>:8080/playground/index.html

Please refer to the
[tensor intro](http://docs.vespa.ai/documentation/tensor-intro.html)
and
[tensor user guide](http://docs.vespa.ai/documentation/tensor-user-guide.html)
for more information.


### Executable example
**Check-out, compile and run:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/basic-search-tensor &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/basic-search-tensor/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
**Feed data into application:**
<pre data-test="exec">
$ curl -s -X POST --data-binary @${VESPA_SAMPLE_APPS}/basic-search-tensor/music-data-1.json \
    http://localhost:8080/document/v1/music/music/docid/1 | python -m json.tool
$ curl -s -X POST --data-binary @${VESPA_SAMPLE_APPS}/basic-search-tensor/music-data-2.json \
    http://localhost:8080/document/v1/music/music/docid/2 | python -m json.tool
</pre>
**Test the application:**
<pre data-test="exec" data-test-assert-contains="Michael Jackson">
$ curl -s 'http://localhost:8080/search/?query=sddocname:music&amp;tensor=%7B%7Bx%3A0%7D%3A1.0%2C%7Bx%3A1%7D%3A2.0%2C%7Bx%3A2%7D%3A3.0%2C%7Bx%3A3%7D%3A5.0%7D' | python -m json.tool
</pre>
**Shutdown and remove the container:**
<pre data-test="after">
$  docker rm -f vespa
</pre>
