<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - transformers

This sample application is a small example of using transformers for ranking
using a small sample from the MS MARCO data set.

**Clone the sample:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ APP_DIR=`pwd`/sample-apps/transformers
$ cd $APP_DIR
</pre>

**Install required packages**:

<pre data-test="exec">
$ pip3 install -qqq --upgrade pip
$ pip3 install -qqq torch transformers
</pre>

**Set up the application package**:

This downloads the transformer model, converts it to an ONNX model and
puts it in the `models` directory. Additionally, since we use this model
for sequence classification, we need to extract two extra tensors for a
linear model on top of the transformer and add them as constants. For
this sample we use a fairly small model:

<pre data-test="exec">
$ MODEL_NAME="nboost/pt-tinybert-msmarco"
$ ./bin/setup-ranking-model.sh $MODEL_NAME
</pre>

**Create data feed:**

Convert from MS MARCO to a Vespa feed. Here we extract from sample data.
To use the entire MS MARCO data set, use the download script. For this
sample application, we feed in pre-tokenized documents, meaning that
tokenization is done outside of Vespa.

<pre data-test="exec">
$ ./bin/convert-msmarco.sh $MODEL_NAME
</pre>

**Start Vespa:**

<pre data-test="exec">
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $APP_DIR:/app --publish 8080:8080 vespaengine/vespa
</pre>

**Wait for the configserver to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

**Deploy the application:**

<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /app/src/main/application && \
    /opt/vespa/bin/vespa-deploy activate'
</pre>

**Wait for the application to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Feed data:**

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /app/msmarco/vespa.json --host localhost --port 8080'
</pre>

**Test the application:**

This script reads from the MS MARCO queries and tokenizes the
queries based on the transformer model and uses the tokens to
query Vespa:

<pre data-test="exec" data-test-assert-contains="children">
$ ./src/python/evaluate.py $MODEL_NAME
</pre>

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>

