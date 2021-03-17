# Vespa Documentation Search
Vespa Documentation Search is a Vespa Cloud instance for searching documents in
* vespa.ai
* cloud.vespa.ai
* blog.vespa.ai
* Vespa Sample applications README files

This sample app is auto-deployed to Vespa Cloud,
see [deploy-vespa-documentation-search.yaml](https://github.com/vespa-engine/sample-apps/blob/master/.github/workflows/deploy-vespa-documentation-search.yaml)

![Vespa-Documentation-Search-Architecture](img/Vespa-Documentation-Search-Architecture.svg)


## Query API
Open API endpoints:
* https://doc-search.vespa.oath.cloud/document/v1/
* https://doc-search.vespa.oath.cloud/search/

Example queries:
* https://doc-search.vespa.oath.cloud/document/v1/open/doc/docid/open%2Fen%2Freference%2Fquery-api-reference.html
* https://doc-search.vespa.oath.cloud/search/?yql=select+*+from+doc+where+userInput(@input)%3B&input=vespa+ranking+is+great

Using these endpoints is a good way to get started with Vespa -
see the [github deploy action](https://github.com/vespa-engine/sample-apps/blob/master/.github/workflows/deploy-vespa-documentation-search.yaml)
(use `vespa:deploy` to deploy to a dev instance or the [quick-start](https://docs.vespa.ai/en/vespa-quick-start.html))
to deploy using Docker.

Generate the `open_index.json` feed file: `cd vespa-engine/documentation && bundle exec jekyll build` -
use this to feed a local instance for experimentation.
Refer to the [vespa_index_generator.rb](https://github.com/vespa-engine/documentation/blob/master/_plugins/vespa_index_generator.rb)
for how to generate the feed file.



## Ranking experiments
Learn how [ranking](https://docs.vespa.ai/en/ranking.html) works in Vespa
by using the open [query API](http://docs.vespa.ai/en/query-api.html).
Ranking is the user-defined computation that scores documents to a query,
here configured in the schema [doc.sd](src/main/application/schemas/doc.sd),
also the [schema documentation](https://docs.vespa.ai/en/schemas.html).
This schema has a set of (contrived) ranking functions, to help learn Vespa ranking


### Ranking using document features
Let's start with something simple: _Irrespective of the query, score documents by the number of in-links to it_.
That is, for any query, return the documents with most in-links first in the result set:

```
    rank-profile inlinks {
        first-phase {
            expression: attribute(inlinks).count
        }
        summary-features {
            attribute(inlinks).count
        }
    }
```

Here we use only the first-phase ranking (Vespa has first- and second-phase) -
the score, named _relevance_ in query results, is the size of the inlinks attribute array in the document:

https://doc-search.vespa.oath.cloud/search/?yql=select%20*%20from%20sources%20*%20where%20sddocname%20contains%20%22doc%22%3B&ranking=inlinks

Count the number of entries in _inlinks_ in the result and compare with _relevance_ - it will be the same.

Observe that the ranking function does not use any features from the query.


### Observing values used in ranking
When developing an application, it is useful to observe the input values to ranking.
Use [summary-features](https://docs.vespa.ai/en/reference/schema-reference.html#summary-features)
to output values in search results.

In this experiment, we will use a more sophisticated rank function, scoring older documents lower.
Also new is:
* use of the _now_ [ranking feature](https://docs.vespa.ai/en/reference/rank-features.html)
* use of constants and functions to write better code

```
    rank-profile inlinks_age {
        first-phase {
            expression: rank_score
        }
        summary-features {
            attribute(inlinks).count
            attribute(last_updated)
            now
            doc_age_seconds
            age_decay
            num_inlinks
            rank_score
        }
        constants {
            decay_const: 0.9
        }
        function doc_age_seconds() {
            expression: now - attribute(last_updated)
        }
        function age_decay() {
            expression: pow(decay_const, doc_age_seconds/3600)
        }
        function num_inlinks() {
            expression: attribute(inlinks).count
        }
        function rank_score() {
            expression: num_inlinks * age_decay
        }
    }
```

In the query results, observe a document with 27 in-links, 9703 seconds old,
get at relevance at 20.32:

```
    "relevance": 20.325190122213748,
...
    "summaryfeatures": {
        "attribute(inlinks).count": 27.0,
        "attribute(last_updated)": 1.615971522E9,
        "now": 1.615981225E9,
        "rankingExpression(age_decay)": 0.7527848193412499,
        "rankingExpression(doc_age_seconds)": 9703.0,
        "rankingExpression(num_inlinks)": 27.0,
        "rankingExpression(rank_score)": 20.325190122213748,
        "vespa.summaryFeatures.cached": 0.0
    }
```

Using _summary-features_ makes it easy to validate and develop the rank expression.


### Two-phased ranking
See [first-phase](https://docs.vespa.ai/en/reference/schema-reference.html#firstphase-rank).
The purpose of two-phased ranking is to use a cheap rank function to eliminate most candidates
using little resources in the first phase -
then use a precise, resource intensive function in the second phase.

```
    rank-profile inlinks_twophase inherits inlinks_age {
        first-phase {
            keep-rank-count       : 50
            rank-score-drop-limit : 10
            expression            : num_inlinks
        }
        second-phase {
            expression            : rank_score
        }
    }
```

Note how using rank-profile inheritance is a smart way to define functions once
and use in multiple rank-profiles.

ToDo: query here

Note in the results that no document has a _rankingExpression(num_inlinks) < 11.0_,
meaning all such documents were purged in the first ranking phase.



## Document feed automation
Vespa Documentation is stored in GitHub:
* https://github.com/vespa-engine/documentation and https://github.com/vespa-engine/frontpage
* https://github.com/vespa-engine/cloud
* https://github.com/vespa-engine/blog
* https://github.com/vespa-engine/sample-apps

Jekyll is used to serve the documentation, it rebuilds at each commit.

A change also triggers GitHub Actions.
The _Build_ step in the workflow uses the Jekyll Generator plugin to build a JSON feed, used in fhe _Feed_ step:
* https://github.com/vespa-engine/documentation/blob/master/.github/workflows/feed.yml
* https://github.com/vespa-engine/documentation/blob/master/_config.yml
* https://github.com/vespa-engine/documentation/blob/master/_plugins/vespa_index_generator.rb



### Security
Vespa Cloud secures endpoints using mTLS. Secrets can be stored in GitHub Settings for a repository.
Here, the private key secret is accessed in the GitHub Actions workflow that feeds to Vespa Cloud:
[feed.yml](https://github.com/vespa-engine/documentation/blob/master/.github/workflows/feed.yml)



## Query integration
Query results are open to the internet. To access Vespa Documentation Search,
an AWS Lambda function is used to get the private key secret from AWS Parameter Store,
then add it to the https request to Vespa Cloud:

The lambda needs AmazonSSMReadOnlyAccess added to its Role to access the Parameter Store.

Note JSON-P being used (_jsoncallback=_) - this simplifies the search result page: [search.html](https://github.com/vespa-engine/documentation/blob/master/search.html).

<!-- ToDo: ref to Vespa JSON interface for this quirk -->



## Vespa Cloud Development and Deployments
This is a Vespa Cloud application and has hence implemented
[automated deployments](https://cloud.vespa.ai/en/automated-deployments).

The feed can contain an array of links from each document.
The [OutLinksDocumentProcessor](src/main/java/ai/vespa/cloud/docsearch/OutLinksDocumentProcessor.java)
is custom java code that add an in-link in each target document using the Vespa Document API.

To test this functionality, the
[VespaDocSystemTest](src/test/java/ai/vespa/cloud/docsearch/VespaDocSystemTest.java) runs for each deployment.

Creating a System Test is also a great way to develop a Vespa application:
* Use this application as a starting point
* Create a Vespa Cloud tenant (i.e. account), and set _tenant_ in [pom.xml](pom.xml)
* Deploy the application to Vespa Cloud
* Run the System Test from maven or IDE using the
  [Endpoint](https://github.com/vespa-engine/vespa/blob/master/tenant-cd-api/src/main/java/ai/vespa/hosted/cd/Endpoint.java)

<!-- ToDo: link to a Vespa Cloud Developer Guide once completed -->



## Status
[![Deploy vespa-documentation-search to Vespa Cloud](https://github.com/vespa-engine/sample-apps/workflows/Deploy%20vespa-documentation-search%20to%20Vespa%20Cloud/badge.svg?branch=master)](https://github.com/vespa-engine/sample-apps/actions?query=workflow%3A%22Deploy+vespa-documentation-search+to+Vespa+Cloud%22)

[![Vespa Documentation Search Feed](https://github.com/vespa-engine/documentation/workflows/Vespa%20Documentation%20Search%20Feed/badge.svg?branch=master)](https://github.com/vespa-engine/documentation/actions?query=workflow%3A%22Vespa+Documentation+Search+Feed%22)

[![Vespa Cloud Documentation Search Feed](https://github.com/vespa-engine/cloud/workflows/Vespa%20Cloud%20Documentation%20Search%20Feed/badge.svg?branch=master)](https://github.com/vespa-engine/cloud/actions?query=workflow%3A%22Vespa+Cloud+Documentation+Search+Feed%22)



## Simplified node.js Lambda code
<pre>
'use strict';
const https = require('https')
const AWS = require('aws-sdk')

const publicCert = `-----BEGIN CERTIFICATE-----
MIIFbDCCA1QCCQCTyf46/BIdpDANBgkqhkiG9w0BAQsFADB4MQswCQYDVQQGEwJO
...
NxoOxvYcP8Pnxn8UGILy7sKl3VRQWIMrlOfXK4DEg8EGqeQzlFVScfSdbH0i6gQz
-----END CERTIFICATE-----`;

exports.handler = async (event, context) => {
    console.log('Received event:', JSON.stringify(event, null, 4));
    const query = event.queryStringParameters.query ? event.queryStringParameters.query : '';
    const jsoncallback = event.queryStringParameters.jsoncallback;
    const path = encodeURI(`/search/?jsoncallback=${jsoncallback}&query=${query}&hits=${hits}&ranking=${ranking}`);

    const ssm = new AWS.SSM();
    const privateKeyParam = await new Promise((resolve, reject) => {
        ssm.getParameter({
            Name: 'ThePrivateKey',
            WithDecryption: true
        }, (err, data) => {
            if (err) { return reject(err); }
            return resolve(data);
        });
    });

    var options = {
        hostname: 'vespacloud-docsearch.vespa-team.aws-us-east-1c.public.vespa.oath.cloud',
        port: 443,
        path: path,
        method: 'GET',
        headers: { 'accept': 'application/json' },
        key: privateKeyParam.Parameter.Value,
        cert: publicCert
    }

    var body = '';
    const response = await new Promise((resolve, reject) => {
        const req = https.get(
            options,
            res => {
                res.setEncoding('utf8');
                res.on('data', (chunk) => {body += chunk})
                res.on('end', () => {
                    resolve({
                        statusCode: 200,
                        body: body
                    });
                });
            });
        req.on('error', (e) => {
          reject({
              statusCode: 500,
              body: 'Something went wrong!'
          });
        });
    });
    return response
};
</pre>
