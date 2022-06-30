<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - Stateless model evaluation

A sample Vespa application to evaluate models of the application package in
Vespa containers.

Please refer to
[stateless model evaluation](https://docs.vespa.ai/en/stateless-model-evaluation.html)
for more information.

The directory `src/main/application/models` contains two ONNX model files generated
by the PyTorch scripts in the same directory. These two models are used to show
various ways stateless model evaluation can be used in Vespa:

- Vespa can automatically make models available through a REST API.
- In a [request handler](https://docs.vespa.ai/en/jdisc/developing-request-handlers.html) providing the capability of
  executing custom code before evaluating a model.
- In searchers and document processors.
- In a post-processing searcher to run a model in batch with the result from the content node.

### Quick Start 

Requirements:
* [Docker](https://www.docker.com/) Desktop installed and running. 6GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64
* Minimum 6GB memory dedicated to Docker (the default is 2GB on macOS).
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [Github releases](https://github.com/vespa-engine/vespa/releases).
* [Java 11](https://openjdk.java.net/projects/jdk/11/) installed.
* [Apache Maven](https://maven.apache.org/install.html) This sample app uses custom Java components and Maven is used
  to build the application.

**Validate environment, should be minimum 6G:**

<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html).

<pre >
$ brew install vespa-cli
</pre>

Set target env, it's also possible to deploy to [Vespa Cloud](https://cloud.vespa.ai/)
using target cloud.

For local deployment using docker image use

<pre data-test="exec">
$ vespa config set target local
</pre>

For cloud deployment using [Vespa Cloud](https://cloud.vespa.ai/) use

<pre>
$ vespa config set target cloud
$ vespa config set application tenant-name.myapp.default
$ vespa auth login 
$ vespa auth cert
</pre>

Where tenant-name is the tenant created when signing up for cloud. 

Pull and start the vespa docker container image:

<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Download this sample application
<pre data-test="exec">
$ vespa clone model-evaluation myapp && cd myapp
</pre>

Build the application package
<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Verify that configuration service (deploy api) is ready

<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Deploy the application 

<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

Wait for the application endpoint to become available

<pre data-test="exec">
$ vespa status --wait 300
</pre>

**Test the application**

Running [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html)
which runs a set of basic tests to verify that the application is working as expected.
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/model-evaluation-test.json
</pre>

**Using the REST APIs directly**
In the following examples we use vespa-cli's curl option, vespa-cli manages the endpoint url, local or cloud. It's 
also possible to switch between the two modes and have a local deployment and a cloud deployment. 

<pre data-test="exec">
$ vespa config set application tenant-name.myapp.default
</pre>

List the available models:
<pre data-test="exec" data-test-assert-contains="transformer">
$ vespa curl /model-evaluation/v1/ 
</pre>

Details of model the `transformer` model:

<pre data-test="exec" data-test-assert-contains="transformer">
$ vespa curl /model-evaluation/v1/transformer
</pre>

Evaluating the model:

<pre data-test="exec" data-test-assert-contains="1.64956">
$ vespa curl -- \
  --data-urlencode "input=[[1,2,3]]" \
  --data-urlencode "format.tensors=short" \
  /model-evaluation/v1/transformer/eval
</pre>

The input here is using [literal short form](https://docs.vespa.ai/en/reference/tensor.html#tensor-literal-form):

**Test the application - Java API in a handler**

<pre data-test="exec" data-test-assert-contains="1.64956">
$ vespa curl -- \
  --data-urlencode "input={{x:0}:1,{x:1}:2,{x:2}:3}" \
  --data-urlencode "model=transformer" \
  /models/
</pre>

The input here is

```
    { {x:0}:1, {x:1}:2, {x:2}:3 }
```

The model expects type `tensor(d0[],d1[])`, but the handler transforms the tensor to the correct shape before evaluating
the model.

**Test the document processor**

Feed in a few documents by first downloading the [vespa-feed-client](https://docs.vespa.ai/en/vespa-feed-client.html): 

<pre data-test="exec">
$ curl -L -o vespa-feed-client-cli.zip \
    https://search.maven.org/remotecontent?filepath=com/yahoo/vespa/vespa-feed-client-cli/7.527.20/vespa-feed-client-cli-7.527.20-zip.zip
$ unzip vespa-feed-client-cli.zip
$ ./vespa-feed-client-cli/vespa-feed-client --verbose --file feed.json --endpoint http://localhost:8080
</pre>

The document processor uses the `transformer` model to generate embeddings that are stored in the content cluster.

**Test the searchers**

<pre data-test="exec" data-test-assert-contains="1.58892">
$ vespa curl -- \
  --data-urlencode "input={{x:0}:1,{x:1}:2,{x:2}:3}" \
  --data-urlencode "searchChain=mychain" \
  /search/
</pre>

This issues a search for the same input as above:

```
    { {x:0}:1, {x:1}:2, {x:2}:3 }
```

The `MySearcher` searcher uses the `transformer` model to translate to an embedding, which is sent to the backend.
A simple dot product between the embeddings of the query and documents are performed, and the documents are initially
sorted in descending order.

The `MyPostProcessingSearcher` uses the `pairwise_ranker` model to compare each document against each other, something
that can't be done on the back end before determining the final rank order.

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
