
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications
For operational sample applications, see [examples/operations](examples/operations). 

## Getting started - Basic Sample Applications

### Basic album-recommendation
The [album-recommendation](album-recommendation/) is the intro application to Vespa.
Learn how to configure the schema for simple recommendation and search use cases.

### Simple hybrid semantic search
The [simple semantic search](simple-semantic-search/)
application demonstrates indexed vector search using `HNSW`, 
creating embedding vectors from a transformer language model inside Vespa, and hybrid
text and semantic ranking. This app also demonstrates using native Vespa embedders. 

### Indexing multiple vectors per field
The [Vespa Multi-Vector Indexing with HNSW](multi-vector-indexing/) demonstrates how to 
index multiple vectors per document field for better semantic search for longer documents.  

### ColBERT token-level embeddings
The [colbert](colbert) application demonstrates how to 
use the Vespa colbert-embedder for explainable semantic search with better accuracy than regular
text embedding models. 

### ColBERT token-level embeddings for long documents
The [colbert-long](colbert-long) application demonstrates how to 
use the Vespa colbert-embedder for explainable semantic search for longer documents. 

### SPLADE sparse learned weights for ranking
The [splade](splade) application demonstrates how to 
use the Vespa splade-embedder for semantic search. 

### Multilingual semantic search
The [multilingual](multilingual-search) sample application demonstrates multilingual semantic search 
with multilingual text embedding models. 

### Customizing embeddings 
The [custom-embeddings](custom-embeddings) application demonstrates customizing frozen document embeddings for downstream tasks. 

## More advanced sample applications

### News search and recommendation tutorial 
The [news](news/) sample application used in the [Vespa tutorial](https://docs.vespa.ai/en/tutorials/news-1-getting-started.html).
This application demonstrates basic search functionality.

It also demonstrates how to build a recommendation system
where the approximate nearest neighbor search in a shared user/item embedding space
is used to retrieve recommended content for a user.
This app also demonstrates using parent-child relationships.

### Billion-scale Image Search
This [billion-scale-image-search](billion-scale-image-search/) app demonstrates 
billion-scale image search using CLIP retrieval. It features separation of compute from storage and query time vector similarity de-duping. PCA dimension reduction and more.

### State-of-the-art Text Ranking
This [msmarco-ranking](msmarco-ranking/) application demonstrates 
how to represent state-of-the-art text ranking using Transformer (BERT) models.
It uses the MS Marco passage ranking datasets and features
bi-encoders, cross-encoders, and late-interaction models (ColBERT).


See also the more simplistic [text-search](text-search) app that demonstrates 
traditional text search using BM25/Vespa nativeRank.

### Next generation E-Commerce Search

The [use-case-shopping](use-case-shopping/) app creates an end-to-end E-Commerce shopping engine.
This use case also bundles a frontend application.
It uses the [Amazon product data set](http://jmcauley.ucsd.edu/data/amazon/links.html).
It demonstrates building next generation E-commerce Search using Vespa. See
also the [commerce-product-ranking](commerce-product-ranking/) sample application for using
learning-to-rank techniques (Including `XGBoost` and `LightGBM`) for improving product search ranking.

### Extractive Question Answering
The [dense-passage-retrieval-with-ann](dense-passage-retrieval-with-ann/) application
demonstrates end-to-end question answering using Facebook's DPR (Dense passage Retriever) model. 
The extractive answering part extracts an answer from the evidence passage(s).

See also [Efficient Open Domain Question Answering with Vespa](https://blog.vespa.ai/efficient-open-domain-question-answering-on-vespa/)
and [Scaling Question Answering with Vespa](https://blog.vespa.ai/from-research-to-production-scaling-a-state-of-the-art-machine-learning-system/).

### Search as you type and query suggestions 
The [incremental-search](incremental-search/) application demonstrates search-as-you-type functionality, where for each keystroke of the user, it retrieves matching documents. 
It also demonstrates search suggestions (query auto-completion).

### Vespa as ML inference server (model-inference)
The [model-inference](model-inference/) application demonstrates 
using Vespa as a stateless ML model inference server
where Vespa takes care of distributing ML models to multiple serving containers,
offering horizontal scaling and safe deployment.
Model versioning and feature processing pipeline.


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

[![sample-apps link checker](https://api.screwdriver.cd/v4/pipelines/7038/link-checker-sample-apps/badge)](https://cd.screwdriver.cd/pipelines/7038/)

[![sample-apps build](https://api.screwdriver.cd/v4/pipelines/7038/build-apps/badge)](https://cd.screwdriver.cd/pipelines/7038/)

[![sample-apps verify-guides](https://api.screwdriver.cd/v4/pipelines/7038/verify-guides/badge)](https://cd.screwdriver.cd/pipelines/7038/)
[![sample-apps verify-guides-big](https://api.screwdriver.cd/v4/pipelines/7038/verify-guides-big/badge)](https://cd.screwdriver.cd/pipelines/7038/)

