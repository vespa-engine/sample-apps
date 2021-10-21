<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - Transformers

This sample application is a small example of using Transformers for ranking
using a small sample from the MS MARCO data set. See also the more comprehensive [MS Marco Ranking sample app](../msmarco-ranking/).


**Validate environment, should be minimum 6G:**
<pre>
$ docker info | grep "Total Memory"
</pre>


**Clone the sample:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/transformers
</pre>


**Install required packages**:

<pre data-test="exec">
$ python3 -m pip install --upgrade pip
$ python3 -m pip install torch transformers onnx onnxruntime
</pre>


**Set up the application package**:

This downloads the transformer model, converts it to an ONNX model and puts it
in the `files` directory. For this sample application, we use a standard
BERT-base model (12 layers, 110 million parameters), however other
[Transformers models](https://huggingface.co/transformers/index.html) can be
used. To export other models, for instance DistilBERT or ALBERT, change the
code in "src/python/setup-model.py". However, this sample application
contains a `WordPiece` tokenizer, so if the Transformer model requires a
different tokenizer, you would have to add that yourself.

<pre data-test="exec">
$ ./bin/setup-ranking-model.sh
</pre>


**Build the application package:**

This sample application contains a Java `WordPiece` tokenizer which is
invoked during document feeding and query handling.
This compiles and  packages the Vespa application:

<pre data-test="exec">
$ mvn clean package
</pre>


**Create data feed:**

Convert from MS MARCO to a Vespa feed. Here we extract from sample data.
To use the entire MS MARCO data set, use the download script.

<pre data-test="exec">
$ ./bin/convert-msmarco.sh
</pre>


**Start Vespa:**

<pre data-test="exec">
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


**Feed data:**

<pre data-test="exec">
$ curl -L -o vespa-http-client-jar-with-dependencies.jar \
    https://search.maven.org/classic/remotecontent?filepath=com/yahoo/vespa/vespa-http-client/7.391.28/vespa-http-client-7.391.28-jar-with-dependencies.jar
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --verbose --file msmarco/vespa.json --endpoint http://localhost:8080
</pre>


**Test the application:**

This script reads from the MS MARCO queries and issues a Vespa query:

<pre data-test="exec" data-test-assert-contains="children">
$ ./src/python/evaluate.py
</pre>


**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
