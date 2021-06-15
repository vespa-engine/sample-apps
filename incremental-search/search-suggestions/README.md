<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample application - search suggestion

Uses streaming search to for document matching.

## Qick start

**Clone sample-apps and go to search-suggestions**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/incremental-search/search-suggestions
</pre>

**Set up and docker container**
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run -m 6G --detach --name vespa --hostname vespa-example \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

**Wait for response** 

Check respons with:
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:19071/ApplicationStatus
</pre>

**Build and deploy the application**
<pre data-test="exec">
$ mvn clean package
$ curl --header Content-Type:application/zip --data-binary @target/application.zip \
  localhost:19071/application/v2/tenant/default/prepareandactivate
</pre>

**Check if application is ready**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Feed the example documents to the application**
<pre data-test="exec">
$ curl -L -o vespa-http-client-jar-with-dependencies.jar \
  https://search.maven.org/classic/remotecontent?filepath=com/yahoo/vespa/vespa-http-client/7.391.28/vespa-http-client-7.391.28-jar-with-dependencies.jar
$ java -jar vespa-http-client-jar-with-dependencies.jar --verbose --file example_query_log.json --endpoint http://localhost:8080
</pre>

**Generating bootstrapped search terms**
<pre data-test="exec">
$ cd ../../../
$ git clone --depth 1 https://github.com/vespa-engine/documentation.git
$ cd documentation/
$ bundle install
$ bundle exec jekyll build -p _plugins-vespafeed
$ cd ../sample-apps/incremental-search/search-suggestions
$ cp ../../../documentation/open_index.json ./
$ python3 count_terms.py open_index.json feed_terms.json 2 top100en.txt
</pre>

**Feeding bootstrapped search terms**
<pre data-test="exec">
$ java -jar vespa-http-client-jar-with-dependencies.jar --verbose --file feed_terms.json --endpoint http://localhost:8080
</pre>

**Check the website and write queries and view suggestions**

Open http://localhost:8080/site/ in a browser.
To validate the site is up:
<pre data-test="exec" data-test-assert-contains="search suggestions">
$ curl -s http://localhost:8080/site/
</pre>

## General

 ### Streaming search

 [Streaming search](https://docs.vespa.ai/documentation/streaming-search.html) matches documents to queries by streaming through them, similar to grep and does not create precomputed indices. With streaming search only raw data of the documents are stored and and term stemming is not supported.

 Streaming search searches through a document collection with a given groupname, so it is important to set a groupname in the id of the documents you intend to use with streaming search.

Streaming search is slower than indexed search, therefore streaming search is most suitable in situations where the document set is relatively small and the documents themself are short.

### The sample application

In this sample application streaming search is used to implement search suggestions based on users' previous queries. By storing user-input as documents you can get some queries that are not suitable for suggestion. A way to combat this is by filtering out user queries that contain terms that are added to a block list. In this sample application a [document processor](https://docs.vespa.ai/en/document-processing.html) is used to filter out such qeries under document feeding. 
