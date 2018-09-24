<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - Basic Search

A sample Vespa application to evaluate models in the application package.

Please refer to
[stateless model evaluation](https://docs.vespa.ai/documentation/stateless-model-evaluation.html)
for more information.

### Executable example

**Check-out, compile and run:**

<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/model-evaluation &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>

**Wait for the configserver to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

**Deploy the application:**

<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/model-evaluation/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>

**Wait for the application to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Test the application - REST API**

<pre data-test="exec" data-test-assert-contains="mnist_softmax">
$ curl -s 'http://localhost:8080/model-evaluation/v1/' | python -m json.tool
</pre>

<pre data-test="exec" data-test-assert-contains="-0.35465">
$ curl -s 'http://localhost:8080/model-evaluation/v1/mnist_softmax/eval?Placeholder=%7B%7Bd1%3A0%7D%3A0.1%7D' | python -m json.tool
</pre>

**Test the application - Java API**

<pre data-test="exec" data-test-assert-contains="-0.35465">
$ curl -s 'http://localhost:8080/models/?model=mnist_softmax&function=default.add&argumentName=Placeholder&argumentValue=%7B%7Bd1%3A0%7D%3A0.1%7D' | python -m json.tool
</pre>

**Shutdown and remove the container:**

<pre data-test="after">
$  docker rm -f vespa
</pre>

