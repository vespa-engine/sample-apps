<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - Multiple bundles

This sample application demonstrates how to build an application that has split
some of its code into a separate bundle.
The extra bundle for this application is found in [multiple-bundles-lib][1] .

Refer to [container components][2] and the [bundle plugin][3] for more information.

Docker requirements same as in the [quick-start](https://docs.vespa.ai/en/vespa-quick-start.html).


### Executable example

**Validate environment, should be minimum 6G:**
<pre>
$ docker info | grep "Total Memory"
</pre>

**Check-out, compile and run:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
</pre>

**Build the "library" bundle:**
<pre data-test="exec">
$ (cd sample-apps/multiple-bundles-lib &amp;&amp; mvn clean install)
</pre>

**Build the main bundle and start the docker container:**
<pre data-test="exec">
$ cd sample-apps/multiple-bundles &amp;&amp; mvn clean verify
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

**Wait for the configserver to start - wait for HTTP/1.1 200 OK:**
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
