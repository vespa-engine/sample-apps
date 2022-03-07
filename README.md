<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications

## Getting started - Basic Sample Applications

### [News search and recommendation tutorial](news)
This is the sample application used in the [Vespa tutorial](https://docs.vespa.ai/en/tutorials/news-1-getting-started.html).
Please follow the tutorial. This application demonstrates basic search functionality.
It also demonstrates how to build a recommendation system
where approximate nearest neighbor search in a shared user/item embedding space
is used to retrieve recommended content for a user.
This sample app also demonstrates use of parent-child relationships.

### [Basic album-recommendation](album-recommendation)
This is the intro application to Vespa.
Learn how to configure the schema for simple recommendation and search use cases.


## Full-fledged State-of-the-Art Search, Ranking and Question Answering applications
These are great starting points for bringing the latest advancements in Search and Ranking to your domain!

### [State-of-the-art-Ranking](https://github.com/vespa-engine/sample-apps/tree/master/msmarco-ranking)
This sample application demonstrates state-of-the-art text ranking
using Transformer (BERT) models and GBDT models for text ranking.
It uses the MS Marco passage and document ranking datasets.

The document ranking part of the sample app uses a trained LTR (Learning to rank) model using GBDT with LightGBM.
The passage ranking part uses multiple state of the art pretrained language models
in a multiphase retrieval and ranking pipeline.
See also [Pretrained Transformer Models for Search](https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-1/) blog post series.
There is also a simpler ranking app also using the MS Marco relevancy dataset.
See [text-search](text-search) which uses traditional IR text matching with BM25/Vespa nativeRank.

### [Next generation E-Commerce Search](use-case-shopping)
Create an end-to-end E-Commerce shopping engine using [use-case-shopping](use-case-shopping).
This use case also bundles a frontend application.
It uses the  [Amazon product data set](http://jmcauley.ucsd.edu/data/amazon/links.html).
It demonstrates building next generation E-commerce Search using Vespa.

### [State of the art Question Answering](dense-passage-retrieval-with-ann)
This sample application demonstrates end to end question answering using Facebook's DPR models
(Dense passage Retriever for Question Answering).
It is using Vespa's approximate nearest neighbor search to efficiently retrieve text passages
from a Wikipedia based collection of 21M passages.
A BERT based reader component reads the top ranking passages and produces the textual answer to the question.
See also [Efficient Open Domain Question Answering with Vespa](https://blog.vespa.ai/efficient-open-domain-question-answering-on-vespa/)
and [Scaling Question Answering with Vespa](https://blog.vespa.ai/from-research-to-production-scaling-a-state-of-the-art-machine-learning-system/).

### [Search as you type and search suggest](incremental-search)
This sample application demonstrates search-as-you-type where for each keystroke of the user,
we retrieve the best matching documents.
It also demonstrates search suggestions (query autocompletion).

## Sample Applications
These sample application demonstrates various Vespa features and capabilities. 

### [Vespa as ML inference server (model-evaluation)](model-evaluation)
A sample Vespa application which demonstrates using Vespa as a stateless ML model inference server
where Vespa takes care of distributing ML models to multiple serving containers,
offering horizontal scaling and safe deployment.
Model versioning and feature processing pipeline.
Stateless ML model serving can also be used in state-of-the-art retrieval and ranking pipelines,
e.g. query classification and encoding text queries to dense vector representation
for efficient retrieval using Vespa's approximate nearest neighbor search.


<!--
[travis](travis)
[part-purchases-demo](part-purchases-demo): A sample Vespa application to assist with with learning how to group according to the [Grouping Guide](https://docs.vespa.ai/en/grouping.html).
[generic-request-processing](generic-request-processing)
http-api-using-*
-->

----

Note: Applications with _pom.xml_ are Java/Maven projects and must be built before being deployed.
Refer to the [Developer Guide](https://docs.vespa.ai/en/developer-guide.html) for more information.

[Contribute](https://github.com/vespa-engine/vespa/blob/master/CONTRIBUTING.md) to the Vespa sample applications.

----

[![Vespa Sampleapps Search Feed](https://github.com/vespa-engine/sample-apps/actions/workflows/feed.yml/badge.svg)](https://github.com/vespa-engine/sample-apps/actions/workflows/feed.yml)

[![sample-apps link checker](https://cd.screwdriver.cd/pipelines/7038/link-checker-sample-apps/badge)](https://cd.screwdriver.cd/pipelines/7038/)

[![sample-apps build](https://cd.screwdriver.cd/pipelines/7038/build-apps/badge)](https://cd.screwdriver.cd/pipelines/7038/)

[![sample-apps verify-guides](https://cd.screwdriver.cd/pipelines/7038/verify-guides/badge)](https://cd.screwdriver.cd/pipelines/7038/)

