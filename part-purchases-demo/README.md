<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - Part Purchases Demo

A sample Vespa application to assist with learning how to group according to the
[Grouping Guide](https://docs.vespa.ai/en/grouping.html).

**Validate environment, should be minimum 6G:**

<pre>
$ docker info | grep "Total Memory"
</pre>


**Check-out, compile and run:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/part-purchases-demo
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
$ tar -C src/main/application -cf - . | gzip | \
  curl --header Content-Type:application/x-gzip --data-binary @- \
  localhost:19071/application/v2/tenant/default/prepareandactivate
</pre>


**Wait for the application to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>


**Generate sample from csv**

<pre>
$ python3 ./parts.py -f purchase.csv > purchase.json
</pre>


**Feed data into application:**

<pre data-test="exec">
$ curl -L -o vespa-feed-client-cli.zip \
    https://search.maven.org/remotecontent?filepath=com/yahoo/vespa/vespa-feed-client-cli/7.527.20/vespa-feed-client-cli-7.527.20-zip.zip
$ unzip vespa-feed-client-cli.zip
$ ./vespa-feed-client-cli/vespa-feed-client \
    --verbose --file purchase.json --endpoint http://localhost:8080
</pre>


**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
