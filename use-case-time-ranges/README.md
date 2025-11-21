<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample applications - efficient time range search

A sample application where documents have widely distributed time stamps,
and you want to efficiently find all documents newer than / older than
some specific date, all documents from a single year, or similar.

For local deployment using Docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespanode \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
</pre>

Verify that configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application:
<pre data-test="exec">
$ vespa clone use-case-time-ranges myapp && cd myapp
</pre>

Build the application package:
<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Deploy the application package:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

Run [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html) -
this runs a set of basic tests to verify that the application is working as expected:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/smoke-test.json
</pre>

Create the data feed:
<pre data-test="exec">
$ mvn exec:java -Dexec.mainClass="ai.vespa.example.timestamps.CreateFeed" -Dexec.classpathScope=test
</pre>

Feed the data:
<pre data-test="exec">
$ vespa feed generated-data.json
</pre>

Test the application:
<pre data-test="exec" data-test-assert-contains="id:item:item::">
$ vespa query "query=1989"
</pre>

Shutdown and remove the container:
<pre data-test="after">
$ docker rm -f vespa
</pre>

----
