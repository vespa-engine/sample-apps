<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Experiment yourself
[CORD-19 Search](https://cord19.vespa.ai/) is built on the [Vespa Cloud](https://cloud.vespa.ai/) service.
To run the same application on your own computer,
use [Vespa.ai](https://vespa.ai/) (open source big data serving engine) and the guide below.

Running the application locally is easy and enables you to play with ranking features.
Use this to make the application more useful for the scientists that needs it,
and feed back improvements to _CORD-19 Search_!

*This document is work in progress, the full procedure will be ready week of March 30*

----

All Vespa Cloud applications can be run locally.
This guides modifies the [cord-19-search](.)
sample application for local deployment using Docker.

Prerequisites:
* [Docker](https://docs.docker.com/engine/installation/) installed
* [Git](https://git-scm.com/downloads) installed
* Operating system: macOS or Linux
* Architecture: x86_64
* *Minimum 10GB* memory dedicated to Docker (the default is 2GB on Macs)

This guide is tested with Docker for Mac, Community Edition-18.06.1-ce-mac73 (26764) and
<em>vespaengine/vespa</em> Docker image built 2020-03-26.

<ol>
<li>
    <p><strong>Validate environment:</strong></p>
<pre>
$ docker info | grep "Total Memory"
</pre>
    <p>Make sure you see something like <em>Total Memory: 9.734GiB</em></p>
</li>

<li>
    <p><strong>Clone the Vespa sample apps from
    <a href="https://github.com/vespa-engine/sample-apps">github</a>:</strong></p>
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
</pre>
</li>

<li>
    <p>Change &lt;nodes&gt;-element two places in
    sample-apps/vespa-cloud/cord-19-search/src/main/application/services.xml
    - use <a href="https://github.com/vespa-engine/sample-apps/blob/master/album-recommendation-selfhosted/src/main/application/services.xml">services.xml</a>
    as reference.
    Copy <a href="https://github.com/vespa-engine/sample-apps/blob/master/album-recommendation-selfhosted/src/main/application/hosts.xml">hosts.xml</a>
    into same location.
    Then build the application:
    </p>
<pre data-test="exec">
$ mvn clean install
</pre>
</li>

<li>
    <p><strong>Start a Vespa Docker container:</strong></p>
<pre data-test="exec">
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
</li>

<li>
    <p><strong>Wait for the configuration server to start - signified by a 200 OK response:</strong></p>
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
</li>

<li>
    <p><strong>Deploy and activate a sample application:</strong></p>
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare \
  /vespa-sample-apps/vespa-cloud/cord-19-search/target/application.zip &amp;&amp; \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
</li>

<li>
    <p><strong>Ensure the application is active - wait for a 200 OK response:</strong></p>
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
</li>

<li>
    <p><strong>Feed documents:</strong></p>
<pre data-test="exec">
$ TBD
</pre>
</li>

<li>
    <p><strong>Make a query:</strong></p>
<pre data-test="exec" data-test-assert-contains="Metallica">
$ TBD
</pre>
</li>

<li>
    <p><strong>Clean up:</strong></p>
<pre data-test="after">
$ docker rm -f vespa
</pre>
</li>

</ol>
