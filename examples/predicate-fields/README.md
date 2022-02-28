<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample application - Predicate Search

Predicate/Boolean Search and how to feed and query is described in
[predicate search](https://docs.vespa.ai/en/predicate-fields.html).

Adding predicate search to an application is easy,
just add a field of type predicate to the .sd file. (Remember to set the arity parameter.)


### Feed and query
1. **Feed** the data:
    ```sh
    $ vespa-feed-client --file adsdata.json --endpoint <endpoint>
    ```

2. **Query** using yql:
    ```sh
    $ curl --data-urlencode 'yql=select * from sources * where predicate(target, {"name":"Wile E. Coyote"},{})' \
      <endpoint>
    ```


### Executable example
**Validate environment, must be minimum 4G:**

Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
for details and troubleshooting:
<pre>
$ docker info | grep "Total Memory"
</pre>

**Check-out, compile and start Docker container:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/examples/predicate-fields &amp;&amp; mvn -U clean package
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
$ curl --header Content-Type:application/zip --data-binary @target/application.zip \
  localhost:19071/application/v2/tenant/default/prepareandactivate
</pre>

**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Feed data into application:**
<pre data-test="exec">
$ curl -L -o vespa-feed-client-cli.zip \
    https://search.maven.org/remotecontent?filepath=com/yahoo/vespa/vespa-feed-client-cli/7.527.20/vespa-feed-client-cli-7.527.20-zip.zip
$ unzip vespa-feed-client-cli.zip
$ ./vespa-feed-client-cli/vespa-feed-client --verbose --file adsdata.json --endpoint http://localhost:8080
</pre>

**Test the application:**
<pre data-test="exec" data-test-assert-contains="ACME Rocket Sled">
$ curl --data-urlencode 'yql=select * from sources * where predicate(target, {"name":"Wile E. Coyote"},{})' \
  http://localhost:8080/search/
</pre>

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
