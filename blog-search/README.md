<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - blog search tutorial

This sample application contains the code for the blog search tutorial.

Please refer to
[Vespa tutorial pt.1 - Blog searching](http://docs.vespa.ai/documentation/tutorials/blog-search.html)
for more information.


**Executable example:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ docker run -m 10G --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/blog-search/src/main/application && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
**Feed data into application:**
<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar --verbose \
  --file /vespa-sample-apps/blog-search/blog-sample-data.json --host localhost --port 8080'
</pre>
**Test the application:**
<pre data-test="exec" data-test-assert-contains="Gerald Finley is passionate about the art of the art song">
$ curl -s 'http://localhost:8080/search/?query=music' | python -m json.tool
</pre>
**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
