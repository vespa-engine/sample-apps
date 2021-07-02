<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample application - search suggestion

Uses indexed prefix search for document matching.

## Qick start

**Clone sample-apps and go to search-suggestions**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/incremental-search/search-suggestions
</pre>

**Set up and run docker container**
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-example \
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
$ java -jar vespa-http-client-jar-with-dependencies.jar --verbose --file example_feed.json --endpoint http://localhost:8080
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

**Generate set of accepted terms**
<pre data-test="exec" data-test-wait-for="200 OK">
$ python3 accepted_words.py open_index.json top100en.txt
</pre>


**Generate terms from query log**

Access logs are assumed to be in *logs/*.
<pre>
$ unzstd -c -r logs/ | grep '"uri":"/search/' | grep 'jsoncallback' \
  | jq '{ term: .uri | scan("(?<=input=)[^&]*") | ascii_downcase | sub("(%..|[^a-z0-9]| )+"; " "; "g") | sub("^ | $"; ""; "g"), hits: .search.hits }' \
  | jq '{update: ("id:term:term::" + (.term | sub(" "; "/"; "g"))), create: true, fields: { term: { assign: .term }, query_count: { increment: 1 }, query_hits: { assign: .hits } } }' > feed_queries.json
</pre>


**Feed terms from query log**
<pre>
$ java -jar vespa-http-client-jar-with-dependencies.jar --verbose --file feed_queries.json --endpoint http://localhost:8080
</pre>

**Check the website and write queries and view suggestions**

Open http://localhost:8080/site/ in a browser.
To validate the site is up:
<pre data-test="exec" data-test-assert-contains="search suggestions">
$ curl -s http://localhost:8080/site/
</pre>

**Shutdown and remove the docker container**

<pre data-test="after">
$ docker rm -f vespa
</pre>


## General

 ### Indexed prefix search

 Indexed prefix search matches documents where the prefix of the hole document matches the string written in the querry.

 To do an indexed prefix search the query has to have \[{"prefix":true}] (see [example](https://docs.vespa.ai/en/streaming-search.html) under match mode) set. 
 It is important to note that this type of prefix search is not supported for fields set to index in the schema. 
 Therefore all feilds you wish to do a prefix search on has to be attriburtes.  

Indexed prefix search is faster than using streaming search, and are also more suitable for situations where multiple concurrent queries might occur and preformance is importante.


### The sample application

In this sample application indexed prefix search is used to implement search suggestions based on users' previous queries. By storing user-input as documents you can get some queries that are not suitable for suggestion. A way to combat this is by filtering out user queries that contain terms that are added to a block list or have a set of accepted terms. In this sample application a [document processor](https://docs.vespa.ai/en/document-processing.html) is used to filter out such qeries under document feeding. For demonstration purposes we have implented both a block list and a set of accepted terms, but usualy only one of these are implemented.
