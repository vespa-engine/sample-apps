<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
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
<pre>
$ git clone https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/vespa-cloud/cord-19-search
</pre>
</li>

<li>
    <p><strong>Generate feed-file.json in current directory:</strong></p>
    <p>Follow procedure in <a href="feeding.md">feeding.md</a></p>
</li>

<li>
    <p><strong>Build the application:</strong></p>
    <p>Change the &lt;nodes&gt;-element in two places in
    <a href="src/main/application/services.xml">src/main/application/services.xml</a>
    - use <a href="https://github.com/vespa-engine/sample-apps/tree/master/album-recommendation/src/main/application/services.xml">services.xml</a>
    as reference.
    Copy <a href="https://github.com/vespa-engine/sample-apps/tree/master/album-recommendation/src/main/application/hosts.xml">hosts.xml</a>
    into same location.
    Then build the application:
    </p>
<pre>
$ mvn clean install
</pre>
</li>

<li>
    <p><strong>Start a Vespa Docker container:</strong></p>
<pre>
$ docker run --detach --name cord19 --hostname vespa-container --privileged \
  --volume $(pwd):/cord-19-search --publish 8080:8080 vespaengine/vespa
</pre>
</li>

<li>
    <p><strong>Wait for the configuration server to start - signified by a 200 OK response:</strong></p>
<pre>
$ docker exec cord19 bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
</li>

<li>
    <p><strong>Deploy and activate a sample application:</strong></p>
<pre>
$ docker exec cord19 bash -c '/opt/vespa/bin/vespa-deploy prepare \
  /cord-19-search/target/application.zip &amp;&amp; \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
</li>

<li>
    <p><strong>Ensure the application is active - wait for a 200 OK response:</strong></p>
<pre>
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
</li>

<li>
    <p><strong>Feed documents:</strong></p>
<pre>
$ docker exec cord19 bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
  --file /cord-19-search/feed-file.json --endpoint http://localhost:8080 --verbose --useCompression'
</pre>
</li>

<li>
    <p><strong>Make a query:</strong></p>
<pre>
$ curl -s http://localhost:8080/search/?query=virus
</pre>
</li>

<li>
    <p><strong>Clean up:</strong></p>
<pre>
$ docker rm -f cord19
</pre>
</li>

</ol>
