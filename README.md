<!-- Copyright 2017 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications

## Featured sample applications
### album-recommendation
This is the intro application to Vespa.
Learn how to configure the schema for simple recommendation and search use cases.
Try the apps on [Vespa Cloud](vespa-cloud) or [using Docker](album-recommendation-selfhosted)

### blog-recommendation
This set of sample apps goes deeper. Starting with [blog-search](blog-search), a more complex search app is build.
Then transform this into an app for making recommendations in [blog-recommendation](blog-recommendation).
Finally, use machine learning to train and use a model for use in recommendations.

### use-case-shopping
Create an end-to-end E-Commerce shopping engine using [use-case-shopping](use-case-shopping)
and an [Amazon product data set](http://jmcauley.ucsd.edu/data/amazon/links.html).

### text-search
[text-search](text-search) dives deep into text ranking, using Vespa's _nativerank_ and BM25 implementations.
It uses the [MS Marco](http://www.msmarco.org/) dataset.

### semantic-qa-retrieval
[semantic-qa-retrieval](semantic-qa-retrieval) takes text search to the next level using the
[Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/),
text embeddings and [tensor ranking](https://docs.vespa.ai/en/reference/tensor.html).
This sample app demonstrates how to return answers to questions.


## Detailed sample applications
[model-evaluation](model-evaluation): A sample Vespa application which demonstrates Stateless ML Model Evaluation.

[boolean-search](boolean-search): Learn how to use predicate fields to implement boolean indexing.
I.e. how to express in a document a range of values to match, like _"this fits readers in age range 20 to 30"_.

[multiple-bundles](multiple-bundles): Build a Java application using components and dependencies in other bundles (jars). 

basic-search-*: Simple application that can be deployed in different environments

<!--
[travis](travis)
[part-purchases-demo](part-purchases-demo): A sample Vespa application to assist with with learning how to group according to the [Grouping Guide](https://docs.vespa.ai/en/grouping.html).
[generic-request-processing](generic-request-processing)
http-api-using-*
-->

----

Note: Applications with _pom.xml_ must be built before being deployed.
Refer to [vespa plugins](https://docs.vespa.ai/en/vespa-plugins.html) for more information.

[Contribute](https://github.com/vespa-engine/vespa/blob/master/CONTRIBUTING.md)
to the Vespa sample applications.

----

Travis-CI build status: [![Build Status](https://travis-ci.com/vespa-engine/sample-apps.svg?branch=master)](https://travis-ci.org/vespa-engine/sample-apps)
