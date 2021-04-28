<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - blog recommendation tutorial

This sample application contains the code for the blog recommendation tutorial. Refer to
[Vespa tutorial pt.2 - Blog recommendation](https://docs.vespa.ai/en/tutorials/blog-recommendation.html).

### Executable example
**Validate environment, should be 10G:**
<pre>
$ docker info | grep "Total Memory"
</pre>

**Check-out, compile and run:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/blog-recommendation &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>
**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:19071/ApplicationStatus
</pre>
**Deploy the application:**
<pre data-test="exec">
$ curl --header Content-Type:application/zip --data-binary @target/application.zip \
  http://localhost:19071/application/v2/tenant/default/prepareandactivate
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-logfmt'  
</pre>
**Really dump the log**
<pre data-test="exec">
$ docker exec vespa bash -c 'cat /opt/vespa/logs/vespa/vespa.log'
</pre>
**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
