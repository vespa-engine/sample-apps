<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - a generic request-response processing application

A simple stateless Vespa application demonstrating general composable request-response processing
with Vespa. No content cluster is configured just a stateless Java container. A custom config class
is created and used to control the processing component.

Please refer to
[developing applications](http://docs.vespa.ai/documentation/jdisc/developing-applications.html)
for more information.


### Executable example
**Check-out, compile and run:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/generic-request-processing &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/generic-request-processing/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
**Test the application:**
<pre data-test="exec" data-test-assert-contains="Hello, services!">
$ curl -s http://localhost:8080/processing/
</pre>
**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
