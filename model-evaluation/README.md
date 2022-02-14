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

### Executable example

**Validate environment, should be minimum 6G:**

<pre>
$ docker info | grep "Total Memory"
</pre>

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

List the available models:

<pre data-test="exec" data-test-assert-contains="transformer">
$ curl -s 'http://localhost:8080/model-evaluation/v1/' | python3 -m json.tool
</pre>

Details of model the `transformer` model:

<pre data-test="exec" data-test-assert-contains="transformer">
$ curl -s 'http://localhost:8080/model-evaluation/v1/transformer' | python3 -m json.tool
</pre>

Evaluating the model:

<pre data-test="exec" data-test-assert-contains="1.64956">
$ curl -s 'http://localhost:8080/model-evaluation/v1/transformer/eval?input=%5B%5B1%2C2%2C3%5D%5D&format=short' | \
  python3 -m json.tool
</pre>

The input here is a URL encoded Vespa tensor in
[literal short form](https://docs.vespa.ai/en/reference/tensor.html#tensor-literal-form):

```
    [[1,2,3]]
```

**Test the application - Java API in a handler**

<pre data-test="exec" data-test-assert-contains="1.64956">
$ curl -s 'http://localhost:8080/models/?model=transformer&input=%7B%7Bx%3A0%7D%3A1%2C%7Bx%3A1%7D%3A2%2C%7Bx%3A2%7D%3A3%7D' | \
  python3 -m json.tool
</pre>

The input here is

```
    { {x:0}:1, {x:1}:2, {x:2}:3 }
```

The model expects type `tensor(d0[],d1[])`, but the handler transforms the tensor to the correct shape before evaluating
the model.

**Test the document processor**

Feed in a few documents by first downloading the `vespa-feed-client` Java client:

<pre data-test="exec">
$ curl -L -o vespa-feed-client-cli.zip \
    https://search.maven.org/remotecontent?filepath=com/yahoo/vespa/vespa-feed-client-cli/7.527.20/vespa-feed-client-cli-7.527.20-zip.zip
$ unzip vespa-feed-client-cli.zip
$ ./vespa-feed-client-cli/vespa-feed-client --verbose --file feed.json --endpoint http://localhost:8080
</pre>

The document processor uses the `transformer` model to generate embeddings that are stored in the content cluster.

**Test the searchers**

<pre data-test="exec" data-test-assert-contains="1.58892">
$ curl -s 'http://localhost:8080/search/?searchChain=mychain&input=%7B%7Bx%3A0%7D%3A1%2C%7Bx%3A1%7D%3A2%2C%7Bx%3A2%7D%3A3%7D' | \
  python3 -m json.tool
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
