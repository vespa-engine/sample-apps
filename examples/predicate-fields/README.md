<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>


# Vespa sample application - Predicate Search

This sample application demonstrates how to use Vespa [predicate fields](https://docs.vespa.ai/en/predicate-fields.html)
for indexing boolean *document* constraints. A predicate is a specification of a
boolean constraint in the form of a boolean expression. Vespa's predicate fields
are used to implement [targeted advertising](https://en.wikipedia.org/wiki/Targeted_advertising)
systems at scale.

For example, this predicate using three target
properties or attributes (not to be confused with Vespa [attributes](https://docs.vespa.ai/en/attributes.html)):

> gender in ['male'] and age in [30..40] and income in [200..50000]

This sample application demonstrates an imaginary two-sided dating marketplace
where users can control visibility in the search or recommendation result page. For example, _Bob_
only wants to be displayed for users that satisfy the following predicate:

> gender in ['male'] and age in [20..40] and hobby in ['climbing', 'sports']

Users who do not satisfy those properties would not be able to see _Bob_'s profile
in the marketplace. Like _Bob_; _Alice_ is picky, she only wants to be shown for males in their thirties with
a high income (measured in thousands).

> gender in ['male'] and age in [30..40] and income in [200..50000]

Both `Bob` and `Alice` are indexed in the marketplace's index system (powered by Vespa of course) as Vespa documents.
The predicate expression in the *document* determines which queries (other users) they would be retrieved for.
The marketplace owner is responsible for managing available targeting properties (e.g `gender`, `age` and `income`) and
at query or recommendation time, set all known properties of the query side user.

We also demonstrate how the marketplace can implement query side filter over regular Vespa fields, so a user `Karen`
can also specify regular query side constraints (for example, searching for users in a certain age group).
This way, the marketplace system has two-sided filtering.
Imagine if deployed ad systems would allow the user to also have constraints on the ads shown,
and not just the other way around?

Finally, we demonstrate how the marketplace can [rank](https://docs.vespa.ai/en/ranking.html) users
using marketplace business metrics like cost-per-click (CPC) and user
[interest embeddings](https://docs.vespa.ai/en/tutorials/news-4-embeddings.html).


## Quick start

Requirements:

* [Docker](https://www.docker.com/) Desktop installed and running. 6 GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* <a href="https://openjdk.org/projects/jdk/17/" data-proofer-ignore>Java 17</a> installed.
* [Apache Maven](https://maven.apache.org/install.html).
  This sample app uses custom Java components and Maven is used to build the application.

Validate Docker resource settings, should be minimum 4 GB:
<pre>
$ docker info | grep "Total Memory"
or
$ podman info | grep "memTotal"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using the docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
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

Deploy the app:
<pre data-test="exec">
$ vespa deploy --wait 300
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started-java#deploy-sample-applications-java).


## Index marketplace users
The users in the imaginary marketplace:
```json lines
{"put": "id:s:user::alice", "fields": {"target": "gender in ['male'] and age in [30..40] and income in [200..50000]", "age": 23, "gender": ["female"]}}
{"put": "id:s:user::bob", "fields": {"target": "gender in ['male'] and age in [20..40] and hobby in ['climbing', 'sports']", "age":41, "gender":["male"]}}
{"put": "id:s:user::karen", "fields": {"target": "gender in ['male'] and age in [30..55]", "age":55, "gender": ["female"]}}
{"put": "id:s:user::mia", "fields": {"target": "gender in ['male'] and age in [50..80]", "age":56,"gender": ["female"]}}
```
`target` is the predicate field, the rest are regular fields.
The `target` predicate field specifies which users the indexed user want's to be shown to,
and the regular fields like `age` and `gender` could be searched by other users in the marketplace.

Feed the documents:

<pre data-test="exec">
$ vespa feed users.jsonl
</pre>

To update one document:

<pre data-test="exec">
$ vespa document -v user.json
</pre>


## Matching using predicate attributes

A user, _Ronald_, enters the marketplace home page and the marketplace knows the following properties about _Ronald_:

- gender: male
- age: 32
- income 3000

The marketplace uses these properties when matching against the index of users using the
[predicate](https://docs.vespa.ai/en/reference/query-language-reference.html#predicate) query operator:

<pre data-test="exec" data-test-assert-contains="alice">
$ vespa query 'yql=select * from sources * where predicate(target, {"gender":["male"]}, {"age":32, "income": 3000})'
</pre>

The above request will retrieve both _Karen_ and _Alice_ as their `target` predicate matches the user properties.

If `Ronald`'s income estimate drops to 100K, _Alice_ will no longer match since _Alice_
has specified a picky income limitation.

<pre data-test="exec" data-test-assert-contains="karen">
$ vespa query 'yql=select * from sources * where predicate(target, {"gender":["male"]}, {"age":32, "income": 100})'
</pre>


## Matching combining predicate with regular filters
Another user, _Jon_, enters the marketplace's search page. The marketplace knows the following properties about _Jon_:

- gender: male
- age: 32
- income 100
- hobby: sports

The marketplace search page will fill in the known properties and perform a search against the index of users:

<pre data-test="exec" data-test-assert-contains="karen">
$ vespa query 'yql=select * from sources * where predicate(target, {"gender":["male"], "hobby":["sports"]}, {"age":32, "income": 100})'
</pre>

The query returns both _Bob_ and _Karen_. Jon is mostly interested in men, so the marketplace can
specify a regular filter on the `gender` field using regular YQL filter syntax, adding `and gender contains "male"`
as a query constraint:

<pre data-test="exec" data-test-assert-contains="bob">
$ vespa query 'yql=select * from sources * where predicate(target, {"gender":["male"], "hobby":["sports"]}, {"age":32, "income": 100}) and gender contains "male"'
</pre>

This is an example of two-sided filtering, both the search user and the indexed user has constraints.


## Matching and ranking

Predicate fields control matching and as we have seen from the above examples,
can also be used with regular query filters.

The combination of document side predicate and query filters determines what documents are returned,
but also which documents (users) are exposed to
[Vespa's ranking framework](https://docs.vespa.ai/en/ranking.html).

Feed data with user profile embedding vectors, and the marketplace business user `cpc`:

<pre data-test="exec">
$ vespa feed users_with_ranking_data.jsonl
</pre>

_Ronald_, enters the marketplace home page again

- gender: male
- age: 32
- income 3000
- Interest embedding representation based on past user to user interactions, or explicit preferences.

And the marketplace runs a recommendation query to display users for _Ronald_:

<pre data-test="exec" data-test-assert-contains="alice">
$ vespa query 'yql=select * from sources * where predicate(target, {"gender":["male"]}, {"age":32, "income": 3000})'
</pre>

Notice that we match both _Alice_ and _Karen_, but _Karen_ is ranked higher because karen has paid more,
her `cpc` score is higher. Notice that the `relevance` is now non-zero, in all the previous examples, the ordering
of the users was non-deterministic. The ranking formula is expressed in the [user](src/main/application/schemas/user.sd)
schema `default` rank-profile

If we now add personalization to the ranking mix, _Alice_ is ranked higher than _Karen_,
as _Alice_ is closer to _Ronald_ in the interest embedding vector space.

This query combines the `predicate` with the [nearestNeighbor](https://docs.vespa.ai/en/nearest-neighbor-search.html)
query operator. The marketplace sends the interest embedding vector representation of _Ronald_ with the query
as a query tensor.

<pre data-test="exec" data-test-assert-contains="alice">
$ vespa query 'yql=select documentid from sources * where (predicate(target, {"gender":["male"]}, {"age":32, "income": 3000})) and ({targetHits:10}nearestNeighbor(profile,profile))' \
  'input.query(profile)=[0.10958350208504841, 0.4642735718813399, 0.7250558657395969, 0.1689946673589695]'
</pre>


### Shutdown and remove the Vespa container
<pre data-test="after">
$ docker rm -f vespa
</pre>
