<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - Building a HTTP API using request handlers and processors

Please refer to
[building a HTTP API using request handlers and processors](http://docs.vespa.ai/documentation/jdisc/http-api-tutorial.html)
for more information.


**Executable example:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/http-api-using-request-handlers-and-processors &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/http-api-using-request-handlers-and-processors/target/application.zip &amp;&amp; \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
<pre data-test="exec" data-test-assert-contains="OK">
$ curl -s 'http://localhost:8080/demo?terms=1%202%203%204'
</pre>
<pre data-test="exec">
$ docker rm -f vespa
</pre>
