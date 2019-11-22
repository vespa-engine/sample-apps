<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - text search tutorial

This sample application contains the code for the text search tutorial. Please refer to the
[text search tutorial](http://docs.vespa.ai/documentation/tutorials/text-search.html)
for more information.

**Executable example:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/text-search &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/apps --publish 8080:8080 vespaengine/vespa
</pre>

**Wait for the configserver to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

**Deploy the application:**

<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /apps/text-search/target/application.zip && \
    /opt/vespa/bin/vespa-deploy activate'
</pre>

**Wait for the application to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Create data feed:** 

To use the entire MS MARCO data set, use the download script. Here we use the sample data. 

<pre data-test="exec">
$ ./bin/convert-msmarco.sh
</pre>

**Feed data:**

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /apps/text-search/msmarco/vespa.json --host localhost --port 8080'
</pre>

**Test the application:**

<pre data-test="exec" data-test-assert-contains="D2977840">
$ curl -s 'http://localhost:8080/search/?query=what+is+dad+bod' 
</pre>

**Browse the site:**

[http://localhost:8080/site](http://localhost:8080/site)

**Install python dependencies:**

<pre data-test="exec">
pip install -r src/python/requirements.txt
</pre>

**Collect training data:**

<pre data-test="exec">
./src/python/collect_training_data.py data collect_rank_features 99
</pre>

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>

