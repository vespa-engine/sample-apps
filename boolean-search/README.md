<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
Predicate Search
==================

Predicate/Boolean Search and how to feed and query is described in
[predicate search](https://docs.vespa.ai/en/predicate-fields.html).

Adding predicate search to an application is easy,
just add a field of type predicate to the .sd file. (Remember to set the arity parameter.)


### Feed and query
1. **Feed** the data:
    ```sh
    java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar --file adsdata.json --endpoint <endpoint>
    ```

2. **Query** using yql, e.g. `select * from sources * where predicate(target, {"name":"Wile E. Coyote"},{});`
    ```sh
    curl "<endpoint url>/search/?yql=select%20*%20from%20sources%20*%20where%20predicate(target%2C%20%7B%22name%22%3A%22Wile%20E.%20Coyote%22%7D%2C%7B%7D)%3B"
    ```


### Executable example
**Validate environment, should be minimum 6G:**
<pre>
$ docker info | grep "Total Memory"
</pre>
**Check-out, compile and start Docker container:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/boolean-search &amp;&amp; mvn clean package
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
$ curl -L -o vespa-http-client-jar-with-dependencies.jar \
    https://search.maven.org/classic/remotecontent?filepath=com/yahoo/vespa/vespa-http-client/7.391.28/vespa-http-client-7.391.28-jar-with-dependencies.jar
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --verbose --file adsdata.xml --endpoint http://localhost:8080
</pre>

**Test the application:**
<pre data-test="exec" data-test-assert-contains="ACME Rocket Sled">
$ curl "http://localhost:8080/search/?yql=select%20*%20from%20sources%20*%20where%20predicate(target%2C%20%7B%22name%22%3A%22Wile%20E.%20Coyote%22%7D%2C%7B%7D)%3B" | python -m json.tool
</pre>

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
