<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - Basic Search

A sample Vespa application to evaluate models in the application package.

Please refer to
[stateless model evaluation](https://docs.vespa.ai/en/stateless-model-evaluation.html)
for more information.

### Executable example

**Check-out, compile and run:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/model-evaluation &amp;&amp; mvn clean package
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

**Test the application - REST API**

<pre data-test="exec" data-test-assert-contains="mnist_softmax">
$ curl -s 'http://localhost:8080/model-evaluation/v1/' | python -m json.tool
</pre>

<pre data-test="exec" data-test-assert-contains="-0.35465">
$ curl -s 'http://localhost:8080/model-evaluation/v1/mnist_softmax/eval?Placeholder=%7B%7Bd0%3A0%2Cd1%3A0%7D%3A0.1%7D' | python -m json.tool
</pre>

**Test the application - Java API**

<pre data-test="exec" data-test-assert-contains="-0.35465">
$ curl -s 'http://localhost:8080/models/?model=mnist_softmax&function=default.add&argumentName=Placeholder&argumentValue=%7B%7Bd0%3A0%2Cd1%3A0%7D%3A0.1%7D' | python -m json.tool
</pre>

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
