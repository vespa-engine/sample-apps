<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - Part Purchases Demo

A sample Vespa application to assist with learning how to group according to the
[Grouping Guide](https://docs.vespa.ai/en/grouping.html).

**Validate environment, should be minimum 4G:**

Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
for details and troubleshooting:
<pre>
$ docker info | grep "Total Memory"
</pre>


**Check-out, compile and run:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/examples/part-purchases-demo
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>


**Wait for the configserver to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:19071/ApplicationStatus
</pre>


**Deploy the application:**

<pre data-test="exec" data-test-assert-contains="prepared and activated.">
$ zip -r - . -x README.md .gitignore "ext/*" "vespa-feed-client-cli*" | \
  curl --header Content-Type:application/zip --data-binary @- \
  localhost:19071/application/v2/tenant/default/prepareandactivate
</pre>


**Wait for the application to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>


**Generate sample from csv**

<pre data-test="exec">
$ python3 ext/parts.py -f ext/purchase.csv > ext/purchase.json
</pre>


**Feed data into application:**

<pre data-test="exec">
$ FEED_CLI_REPO="https://repo1.maven.org/maven2/com/yahoo/vespa/vespa-feed-client-cli" \
	&& FEED_CLI_VER=$(curl -Ss "${FEED_CLI_REPO}/maven-metadata.xml" | sed -n 's/.*&lt;release&gt;\(.*\)&lt;.*&gt;/\1/p') \
	&& curl -SsLo vespa-feed-client-cli.zip ${FEED_CLI_REPO}/${FEED_CLI_VER}/vespa-feed-client-cli-${FEED_CLI_VER}-zip.zip \
	&& unzip -o vespa-feed-client-cli.zip
$ ./vespa-feed-client-cli/vespa-feed-client \
    --verbose --file ext/purchase.json --endpoint http://localhost:8080
</pre>


**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
