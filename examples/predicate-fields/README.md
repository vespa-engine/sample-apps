<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample application - Predicate Search

This sample application demonstrates how to use Vespa predicate fields for indexing boolean expression, 
see also [predicate field documentation](https://docs.vespa.ai/en/predicate-fields.html).

This sample application demonstrates a two-sided marketplace,
where users can specify which other users they want to be matched against.
For example, our alice user, only wants to be displayed for male users,
in a specific age group with an income in a specific high range.
When another user enters the marketplace, searching for other users,
we fill in the known user attributes like gender, age and income,
which is passed with the search the user is doing.

> gender in ['male'] and age in [30..40] and income in [200..50000]

We also combine the user predicate with regular filters, and [ranking](https://docs.vespa.ai/en/ranking.html) 
using business metrics (CPC), and user [interest embeddings](https://docs.vespa.ai/en/tutorials/news-4-embeddings.html). 

## Quick start

Requirements:

* [Docker](https://www.docker.com/) Desktop installed and running. 6 GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* [Java 17](https://openjdk.org/projects/jdk/17/) installed.
* [Apache Maven](https://maven.apache.org/install.html).
  This sample app uses custom Java components and Maven is used to build the application.

Validate Docker resource settings, should be minimum 4 GB:

<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):

<pre >
$ brew install vespa-cli
</pre>

Set target env, it's also possible to deploy to [Vespa Cloud](https://cloud.vespa.ai/)
using target cloud.

For local deployment using the docker image, use:

<pre data-test="exec">
$ vespa config set target local
</pre>

For cloud deployment using [Vespa Cloud](https://cloud.vespa.ai/), use:

<pre>
$ vespa config set target cloud
$ vespa config set application tenant-name.myapp.default
$ vespa auth login 
$ vespa auth cert
</pre>

See also [Cloud Vespa getting started guide](https://cloud.vespa.ai/en/getting-started).
It's possible to switch between local deployment and cloud deployment by changing the `config target`.

Pull and start the vespa docker container image:

<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Verify that configuration service (deploy api) is ready:

<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application:

<pre data-test="exec">
$ vespa clone examples/predicate-fields my-app && cd my-app
</pre>

Build the sample app:

<pre data-test="exec">
$ mvn package -U 
</pre>

Finally, deploy the app:

<pre data-test="exec">
$ vespa deploy 
</pre>

## Index users  
This example uses the [vespa-feed-client](https://docs.vespa.ai/en/vespa-feed-client.html) to feed documents.
The users in the marketplace:
```json lines
{"put": "id:s:user::alice", "fields": {"target": "gender in ['male'] and age in [30..40] and income in [200..50000]", "age": 23, "gender": ["female"]}}
{"put": "id:s:user::bob", "fields": {"target": "gender in ['male'] and age in [20..40] and hobby in ['climbing', 'sports']", "age":41, "gender":["male"]}}
{"put": "id:s:user::karen", "fields": {"target": "gender in ['male'] and age in [30..55]", "age":55, "gender": ["female"]}}
{"put": "id:s:user::mia", "fields": {"target": "gender in ['male'] and age in [50..80]", "age":56,"gender": ["female"]}}
```
`target` is the predicate field, the rest are regular fields.

Download the latest `vespa-feed-client`:
<pre>
$ FEED_CLI_REPO="https://repo1.maven.org/maven2/com/yahoo/vespa/vespa-feed-client-cli" \
  && FEED_CLI_VER=$(curl -Ss "${FEED_CLI_REPO}/maven-metadata.xml" | sed -n 's/.*&lt;release&gt;\(.*\)&lt;.*&gt;/\1/p') \
  && curl -SsLo vespa-feed-client-cli.zip ${FEED_CLI_REPO}/${FEED_CLI_VER}/vespa-feed-client-cli-${FEED_CLI_VER}-zip.zip \
  && unzip -o vespa-feed-client-cli.zip
</pre>

Feed the documents:
<pre data-test="exec">
$ ./vespa-feed-client-cli/vespa-feed-client --verbose \
  --file users.jsonl \
  --endpoint http://localhost:8080
</pre>

When feeding to Vespa cloud endpoints, you also need to include the data plane certificate and key -
[example](https://github.com/vespa-cloud/vector-search#feeding-example).

One can also feed using curl or the Vespa-cli:

<pre data-test="exec">
$ vespa document -v user.json
</pre>

## Matching using predicate attributes

This retrieves both _karen_ and _alice_:
<pre data-test="exec" data-test-assert-contains="alice">
$ vespa query 'yql=select * from sources * where predicate(
  target,
  {"gender":["male"]},
  {"age":32, "income": 3000})'
</pre>

If we change the income to 100, _alice_ will no longer match since _alice_ has specified an income range:

<pre data-test="exec" data-test-assert-contains="karen">
$ vespa query 'yql=select * from sources * where predicate(
  target,
  {"gender":["male"]},
  {"age":32, "income": 100})'
</pre>


## Matching combining predicate with regular filters 
The following query retrieves both _karen_ and _bob_:
<pre data-test="exec" data-test-assert-contains="karen">
$ vespa query 'yql=select * from sources * where predicate(
  target,
  {"gender":["male"], "hobby":["sports"]},
  {"age":32, "income": 100})'
</pre>

We can specify a regular filter on the `gender` field using regular YQL filter syntax:

<pre data-test="exec" data-test-assert-contains="bob">
$ vespa query 'yql=select * from sources * where predicate(
  target,
  {"gender":["male"], "hobby":["sports"]},
  {"age":32, "income": 100})
  and gender contains "male"'
</pre>

This is an example of two-sided filtering, both the search user and the indexed user has constraints. 

## Matching and ranking 

Predicate fields control matching and as we have seen from the above examples,
can also be used with regular query filters.
The combination of document side predicate and query filters determines what documents are returned, 
but also which documents (users) are exposed to [Vespa's ranking framework](https://docs.vespa.ai/en/ranking.html). 

Feed data with user profile embedding vectors,  and `cpc`:

<pre data-test="exec">
$ ./vespa-feed-client-cli/vespa-feed-client --verbose \
  --file users_with_ranking_data.jsonl \
  --endpoint http://localhost:8080
</pre>

Run a query for a male user with high income and age 32:

<pre data-test="exec" data-test-assert-contains="alice">
$ vespa query 'yql=select * from sources * where predicate(
  target,
  {"gender":["male"]},
  {"age":32, "income": 3000})'
</pre>

Notice that we match both alice and karen, but karen is ranked higher because karen has paid more,
her `cpc` score is higher.

If we now add personalization to the ranking mix, _alice_ is ranking higher,
as she matches the query user interests perfectly:

<pre data-test="exec" data-test-assert-contains="alice">
$ vespa query 'yql=select documentid from sources * where (predicate(
  target,
  {"gender":["male"]},
  {"age":32, "income": 3000}))
  and ({targetHits:10}nearestNeighbor(profile,profile))' \
  'input.query(profile)=[0.10958350208504841, 0.4642735718813399, 0.7250558657395969, 0.1689946673589695]'
</pre>

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
