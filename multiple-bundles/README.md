<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - Multiple bundles

This sample application demonstrates how to build an application that has split
some of its code into a separate bundle.
The extra bundle for this application is found in [multiple-bundles-lib][1] .

Refer to [container components][2] and the [bundle plugin][3] for more information.

Docker requirements same as in the [quick-start](https://docs.vespa.ai/en/vespa-quick-start.html).


### Executable example
**Check-out, compile and run:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APPS=`pwd`/sample-apps
</pre>

**Build the "library" bundle:**
<pre data-test="exec">
$ cd $VESPA_SAMPLE_APPS/multiple-bundles-lib &amp;&amp; mvn clean install
</pre>

**Build the main bundle and start the docker container:**
<pre data-test="exec">
$ cd $VESPA_SAMPLE_APPS/multiple-bundles &amp;&amp; mvn clean verify
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>


**Wait for the configserver to start - wait for HTTP/1.1 200 OK:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/multiple-bundles/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>

**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Test the application:**
<pre data-test="exec" data-test-assert-contains="fib(0) = 0">
$ curl -s http://localhost:8080/search/
</pre>
Sending multiple requests generates the Fibonacci number sequence in the 'message' field
of the search result.

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>

[1]: https://github.com/vespa-engine/sample-apps/tree/master/multiple-bundles-lib
[2]: https://docs.vespa.ai/en/jdisc/container-components.html
[3]: https://docs.vespa.ai/en/bundle-plugin.html
