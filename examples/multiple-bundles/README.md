<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Multiple bundles

This sample application demonstrates how to build an application that has split
some of its code into a separate bundle.
The extra bundle for this application is found in [multiple-bundles-lib](../multiple-bundles-lib) .

Refer to [container components](https://docs.vespa.ai/en/jdisc/container-components.html)
and the [bundle plugin](https://docs.vespa.ai/en/bundle-plugin.html) for more information.

Docker requirements same as in the [quick-start](https://docs.vespa.ai/en/vespa-quick-start.html).


### Executable example

**Validate environment, must be minimum 4G:**

Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
for details and troubleshooting:
<pre>
$ docker info | grep "Total Memory"
or
$ podman info | grep "memTotal"
</pre>

**Check-out, compile and run:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
</pre>

**Build the "library" bundle:**
<pre data-test="exec">
$ (cd sample-apps/examples/multiple-bundles-lib &amp;&amp; mvn clean install)
</pre>

**Build the main bundle and start the docker container:**
<pre data-test="exec">
$ cd sample-apps/examples/multiple-bundles &amp;&amp; mvn clean verify
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
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
<pre data-test="exec" data-test-assert-contains="fib(">
$ curl -s http://localhost:8080/search/
</pre>
Sending multiple requests generates the Fibonacci number sequence in the 'message' field
of the search result.

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
