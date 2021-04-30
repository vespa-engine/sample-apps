<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
Predicate Search
==================

Predicate/Boolean Search and how to feed and query is described in
[predicate search](https://docs.vespa.ai/en/predicate-fields.html).

Adding predicate search to an application is easy. Just add a field of
type predicate to the .sd file. (Remember to set the arity parameter.)


### Feed and search
1. **Feed** the data that is to be searched:
    ```sh
    java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar --file adsdata.xml --host <endpoint-host> --port 8080
    ```

2. **Search** using yql expressions, e.g. `select * from sources * where predicate(target, {"name":"Wile E. Coyote"},{});`
    ```sh
    curl "<endpoint url>/search/?query=sddocname:ad&yql=select%20*%20from%20sources%20*%20where%20predicate(target%2C%20%7B%22name%22%3A%22Wile%20E.%20Coyote%22%7D%2C%7B%7D)%3B"
    ```


### Executable example
**Validate environment, should be minimum 6G:**
<pre>
$ docker info | grep "Total Memory"
</pre>
**Check-out, compile and run:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/boolean-search &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/boolean-search/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
**Feed data into application:**
<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar --verbose --file /vespa-sample-apps/boolean-search/adsdata.xml --host localhost --port 8080'
</pre>
**Test the application:**
<pre data-test="exec" data-test-assert-contains="ACME Rocket Sled">
$ curl "http://localhost:8080/search/?query=sddocname:ad&amp;yql=select%20*%20from%20sources%20*%20where%20predicate(target%2C%20%7B%22name%22%3A%22Wile%20E.%20Coyote%22%7D%2C%7B%7D)%3B" | python -m json.tool
</pre>
**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
