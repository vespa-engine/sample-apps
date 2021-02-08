<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - News recommendation

**Clone the sample:**

<pre data-test="exec">
$ # git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ mkdir sample-apps
$ cp -r /Users/lesters/github/sample-apps/news-recommendation sample-apps/
$ APP_DIR=`pwd`/sample-apps/news-recommendation
$ cd $APP_DIR
</pre>

**Install required packages**:

<pre data-test="exec">
$ pip3 install -qqq --upgrade pip
$ pip3 install -qqq -r requirements.txt
</pre>

**Download demo dataset**:

<pre data-test="exec">
$ ./bin/download-mind.sh demo
</pre>

**Convert to Vespa format**:

<pre data-test="exec">
$ ./bin/convert-mind.sh mind
</pre>

**Build the application package:**

<pre data-test="exec">
$ mvn clean package
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
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /app/target/application.zip && \
    /opt/vespa/bin/vespa-deploy activate'
</pre>

**Wait for the application to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Feed data:**

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /app/mind/vespa.json --host localhost --port 8080'
</pre>

**Create user and news embeddings**:

Only run for 5 epochs for now:

<pre data-test="exec">
$ ./src/python/train_gmf.py mind 5
</pre>

**Convert embeddings to Vespa format**:

Adds 0.0 for user embeddings, and max length for news embeddings. 
See https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

<pre data-test="exec">
$ ./bin/convert-embeddings.sh
</pre>

**Feed embeddings**:

News is a partial update as they are already written:

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /app/mind/vespa_user_embeddings.json --host localhost --port 8080'
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /app/mind/vespa_news_embeddings.json --host localhost --port 8080'
</pre>


**Test the application:**

TODO

<pre data-test="exec">
$ curl "http://localhost:8080/search/?searchChain=user&user_id=U82271"
</pre>

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>

