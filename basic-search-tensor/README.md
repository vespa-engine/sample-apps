<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - album recommendations
Vespa is used for online Big Data serving, which means ranking (large) data sets using query data.
Below is an example of how to rank music albums using a user profile -
match albums with scores for a set of categories with a user's preference:

**User profile**

    {
      { cat:pop  }: 0.8,
      { cat:rock }: 0.2,
      { cat:jazz }: 0.1
    }

 **Albums**

    {
      "fields": {
        "album": "A Head Full of Dreams",
        "artist": "Coldplay",
        "year": 2015,
        "category_scores": {
          "cells": [
            { "address": { "cat": "pop"},  "value": 1  },
            { "address": { "cat": "rock"}, "value": 0.2},
            { "address": { "cat": "jazz"}, "value": 0  }
          ]
        }
      }
    }

    {
      "fields": {
        "album": "Love Is Here To Stay",
        "artist": "Diana Krall",
        "year": 2018,
        "category_scores": {
          "cells": [
            { "address": { "cat": "pop" },  "value": 0.4 },
            { "address": { "cat": "rock" }, "value": 0   },
            { "address": { "cat": "jazz" }, "value": 0.8 }
          ]
        }
      }
    }

    {
      "fields": {
        "album": "Hardwired...To Self-Destruct",
        "artist": "Metallica",
        "year": 2016,
        "category_scores": {
          "cells": [
            { "address": { "cat": "pop" },  "value": 0 },
            { "address": { "cat": "rock" }, "value": 1 },
            { "address": { "cat": "jazz" }, "value": 0 }
          ]
        }
      }
    }

**Rank profile**

A [rank profile](https://docs.vespa.ai/documentation/ranking.html) calculates a _relevance score_ per document.
This is defined by the application author - in this case, it is the tensor product -
The data above is represented using [tensors](http://docs.vespa.ai/documentation/tensor-intro.html)..
As the tensor is one-dimensional (the _cat_ dimension), this a vector,
hence this is the dot product of the user profile and album categories:

    rank-profile rank_albums inherits default {
        first-phase {
            expression: sum(query(user_profile) * attribute(category_scores))
        }
    }

Hence, the expected scores are:
<table>
<tr><th>Album</th>                                   <th>pop</th>     <th>rock</th>    <th>jazz</th>      <th>total</th></tr>
<tr><td>A Head Full of Dreams</td>         <td>0.8*1.0</td><td>0.2*0.2</td><td>0.1*0.0</td><td>0.84</td></tr>
<tr><td>Love Is Here To Stay</td>            <td>0.8*0.4</td><td>0.2*0.0</td><td>0.1*0.8</td><td>0.4</td></tr>
<tr><td>Hardwired...To Self-Destruct</td><td>0.8*0.0</td><td>0.2*1.0</td><td>0.1*0.0</td><td>0.2</td></tr>
</table>

Build and test the application, and validate that the document's _relevance_ is the expected value,
and the results are returned in descending relevance order.


### Executable example
**Check-out, compile and run:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/basic-search-tensor &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/basic-search-tensor/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
**Feed data:**
<pre data-test="exec">
$ curl -s -X POST --data-binary @${VESPA_SAMPLE_APPS}/basic-search-tensor/src/test/resources/A-Head-Full-of-Dreams.json \
    http://localhost:8080/document/v1/music/music/docid/1 | python -m json.tool
$ curl -s -X POST --data-binary @${VESPA_SAMPLE_APPS}/basic-search-tensor/src/test/resources/Love-Is-Here-To-Stay.json \
    http://localhost:8080/document/v1/music/music/docid/2 | python -m json.tool
$ curl -s -X POST --data-binary @${VESPA_SAMPLE_APPS}/basic-search-tensor/src/test/resources/Hardwired...To-Self-Desctruct.json \
    http://localhost:8080/document/v1/music/music/docid/3 | python -m json.tool
</pre>
**Recommend albums, send user profile in query:**
<pre data-test="exec" data-test-assert-contains="Metallica">
$ curl -s 'http://localhost:8080/search/?ranking=rank_albums&amp;yql=select%20%2A%20from%20sources%20%2A%20where%20sddocname%20contains%20%22music%22%3B&amp;ranking.features.query(user_profile)=%7B%7Bcat%3Apop%7D%3A0.8%2C%7Bcat%3Arock%7D%3A0.2%2C%7Bcat%3Ajazz%7D%3A0.1%7D' | python -m json.tool
</pre>
**Shutdown and remove the container:**
<pre data-test="after">
$  docker rm -f vespa
</pre>


<hr />

Also includes the rank expression playground used to visualize rank operations. This sample application is based on [basic-search-java](https://github.com/vespa-engine/sample-apps/tree/master/basic-search-java) which requires building the application using maven before deploying. See [Developing application](http://docs.vespa.ai/documentation/jdisc/developing-applications.html). Once deployed you can view the tensor playground:

    http://<host>:8080/playground/index.html
