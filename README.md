<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>


# Vespa sample applications
First-time users should go through the [getting-started](https://docs.vespa.ai/en/getting-started.html) guides first.

The [Vespa](https://vespa.ai/) sample applications are created to run both self-hosted and on Vespa Cloud.
You can easily deploy the sample applications to Vespa Cloud without changing the files -
just follow the same steps as for [vector-search](https://github.com/vespa-cloud/vector-search),
adding security credentials.

For operational sample applications, see [examples/operations](examples/operations).



## Getting started
The [album-recommendation](album-recommendation/) is the intro application to Vespa.
Learn how to configure the schema for simple recommendation and search use cases.

[Pyvespa: Hybrid Search - Quickstart](https://pyvespa.readthedocs.io/en/latest/getting-started-pyvespa.html) and
[Pyvespa: Hybrid Search - Quickstart on Vespa Cloud](https://pyvespa.readthedocs.io/en/latest/getting-started-pyvespa-cloud.html)
create a hybrid text search application combining traditional keyword matching with semantic vector search (dense retrieval).
They also demonstrate the Vespa native embedder functionality.
These are intro level applications for Python users using more advanced Vespa features.
Use [Pyvespa: Authenticating to Vespa Cloud](https://pyvespa.readthedocs.io/en/latest/authenticating-to-vespa-cloud.html) for Vespa Cloud credentials.

[Pyvespa: Querying Vespa](https://pyvespa.readthedocs.io/en/latest/query.html)
is a good start for Python users, exploring how to query Vespa using the Vespa Query Language (YQL).

[Pyvespa: Read and write operations](https://pyvespa.readthedocs.io/en/latest/reads-writes.html)
documents ways to feed, get, update and delete data;
Using context manager with with for efficiently managing resources
and feeding streams of data using `feed_iter` which can feed from streams, Iterables, Lists
and files by the use of generators.

[Pyvespa: Application packages](https://pyvespa.readthedocs.io/en/latest/application-packages.html)
is a good intro to the concept of application packages in Vespa.
Try [Advanced Configuration](https://pyvespa.readthedocs.io/en/latest/advanced-configuration.html) for Vespa Services configuration.

[Pyvespa: Examples](https://pyvespa.readthedocs.io/en/latest/examples/pyvespa-examples.html)
is a repository of small snippets and examples, e.g. really simple vector distance search applications.



## Vector Search, Hybrid Search and Embeddings
There is a growing interest in AI-powered vector representations of unstructured multimodal data
and searching efficiently over these representations.
[vector-search](https://github.com/vespa-cloud/vector-search)
describes how to unlock the full potential of multimodal AI-powered vector representations using Vespa Cloud -
the industry-leading managed Vector Search Service.

The [simple semantic search](simple-semantic-search/)
application demonstrates indexed vector search using `HNSW`,
creating embedding vectors from a transformer language model inside Vespa, and hybrid text and semantic ranking.
This app also demonstrates using native Vespa embedders.

The [Vespa Multi-Vector Indexing with HNSW](multi-vector-indexing/) /
[Pyvespa: Multi-vector indexing with HNSW](https://pyvespa.readthedocs.io/en/latest/examples/multi-vector-indexing.html)
applications demonstrate how to index multiple vectors per document field for semantic search for longer documents.

The [vector-streaming-search](vector-streaming-search) app
demonstrates how to use vector streaming search for naturally partitioned data.
See also [blog post](https://blog.vespa.ai/announcing-vector-streaming-search/).

The [colbert](colbert) application (simple hybrid search with ColBERT) demonstrates how to
use the Vespa [colbert-embedder](https://docs.vespa.ai/en/embedding.html#colbert-embedder)
for explainable semantic search with better accuracy than regular text embedding models.

The [multilingual](multilingual-search) sample application demonstrates multilingual semantic search
with multilingual text embedding models.

ColBERT token-level embeddings for long documents;
The [colbert-long](colbert-long) application demonstrates how to
use the Vespa [colbert-embedder](https://docs.vespa.ai/en/embedding.html#colbert-embedder)
for explainable semantic search for longer documents.

SPLADE sparse learned weights for ranking;
The [splade](splade) application demonstrates how to
use the Vespa [splade-embedder](https://docs.vespa.ai/en/embedding.html#splade-embedder) for
semantic search using sparse vector representations.

[custom-embeddings](custom-embeddings)
demonstrates customizing frozen document embeddings for downstream tasks.

[Pyvespa: Billion-scale vector search with Cohere binary embeddings in Vespa](https://pyvespa.readthedocs.io/en/latest/examples/billion-scale-vector-search-with-cohere-embeddings-cloud.html)
demonstrates using the [Cohere int8 & binary Embeddings](https://cohere.com/blog/int8-binary-embeddings)
with a coarse-to-fine search and re-ranking pipeline that reduces costs, but offers the same retrieval (nDCG) accuracy.
The packed binary vector representation is stored in memory,
with an optional [HNSW index](https://docs.vespa.ai/en/approximate-nn-hnsw.html) using
[hamming](https://docs.vespa.ai/en/reference/schema-reference.html#hamming) distance.
The `int8` vector representation is stored on disk
using Vespa’s [paged](https://docs.vespa.ai/en/attributes.html#paged-attributes) option.

[Pyvespa: BGE-M3 - The Mother of all embedding models](https://pyvespa.readthedocs.io/en/latest/examples/mother-of-all-embedding-models-cloud.html).
This notebook demonstrates how to use the [BGE-M3](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/BGE_M3.pdf) embeddings
and represent all three embedding representations in Vespa!
Vespa is the only scalable serving engine that can handle all M3 representations.
This code is inspired by the README from the model hub [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3).

[Pyvespa: Evaluating retrieval with Snowflake arctic embed](https://pyvespa.readthedocs.io/en/latest/examples/evaluating-with-snowflake-arctic-embed.html).
demonstrates how different rank profiles in Vespa can be set up and evaluated.
For the rank profiles that use semantic search,
we will use the small version of [Snowflake’s arctic embed model series](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) for generating embeddings.

[Pyvespa: Using Cohere Binary Embeddings in Vespa](https://pyvespa.readthedocs.io/en/latest/examples/cohere-binary-vectors-in-vespa-cloud.html)
demonstrates how to use the Cohere binary vectors with Vespa,
including a re-ranking phase that uses the float query vector version for improved accuracy.

[Pyvespa: Multilingual Hybrid Search with Cohere binary embeddings and Vespa](https://pyvespa.readthedocs.io/en/latest/examples/multilingual-multi-vector-reps-with-cohere-cloud.html).
This notebook demonstrates:
* Building a multilingual search application over a sample of the German split of Wikipedia using
  [binarized Cohere embeddings](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary).
* Indexing multiple binary embeddings per document; without having to split the chunks across multiple retrievable units.
* Hybrid search, combining the lexical matching capabilities of Vespa with Cohere binary embeddings.
* Re-scoring the binarized vectors for improved accuracy.

[Pyvespa: Exploring the potential of OpenAI Matryoshka 🪆 embeddings with Vespa](https://pyvespa.readthedocs.io/en/latest/examples/Matryoshka_embeddings_in_Vespa-cloud.html)
demonstrates the effectiveness of using the recently released (as of January 2024) OpenAI `text-embedding-3` embeddings with Vespa.
Specifically, we are interested in the [Matryoshka Representation Learning](https://aniketrege.github.io/blog/2024/mrl/) technique used in training,
which lets us “shorten embeddings (i.e. remove some numbers from the end of the sequence) without the embedding losing its concept-representing properties”.
This allow us to trade off a small amount of accuracy in exchange for much smaller embedding sizes,
so we can store more documents and search them faster.

[Pyvespa: Using Mixedbread.ai embedding model with support for binary vectors](https://pyvespa.readthedocs.io/en/latest/examples/mixedbread-binary-embeddings-with-sentence-transformers-cloud.html)
demonstrates how to use the Mixedbread [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) model
with support for binary vectors with Vespa.
The notebook example also includes a re-ranking phase that uses the float query vector version for improved accuracy.
The re-ranking step makes the model perform at 96.45% of the full float version,
with a 32x decrease in storage footprint.



## Retrieval Augmented Generation (RAG) and Generative AI
The [retrieval-augmented-generation](retrieval-augmented-generation) sample application
demonstrates how to build an end-to-end RAG pipeline with API-based and local LLMs.

[Pyvespa: Turbocharge RAG with LangChain and Vespa Streaming Mode for Partitioned Data](https://pyvespa.readthedocs.io/en/latest/examples/turbocharge-rag-with-langchain-and-vespa-streaming-mode-cloud.html)
illustrates using [Vespa streaming mode](https://docs.vespa.ai/en/streaming-search.html)
to build cost-efficient RAG applications over naturally sharded data.
This notebook is also available as a blog post:
[Turbocharge RAG with LangChain and Vespa Streaming Mode for Sharded Data](https://blog.vespa.ai/turbocharge-rag-with-langchain-and-vespa-streaming-mode/).

[Pyvespa: Visual PDF RAG with Vespa - ColPali demo application](https://pyvespa.readthedocs.io/en/latest/examples/visual_pdf_rag_with_vespa_colpali_cloud.html).
We created an end-to-end demo application for visual retrieval of PDF pages using Vespa, including a frontend web application.
To see the live demo, visit [vespa-engine-colpali-vespa-visual-retrieval.hf.space](https://vespa-engine-colpali-vespa-visual-retrieval.hf.space/).
The main goal of the demo is to make it easy for you to create your own PDF Enterprise Search application using Vespa.

[Pyvespa: Chat with your pdfs with ColBERT, LangChain, and Vespa](https://pyvespa.readthedocs.io/en/latest/examples/chat_with_your_pdfs_using_colbert_langchain_and_Vespa-cloud.html)
illustrates using [Vespa streaming mode](https://docs.vespa.ai/en/streaming-search.html)
to build cost-efficient RAG applications over naturally sharded data.
It also demonstrates how you can now use ColBERT ranking natively in Vespa,
which can now handle the ColBERT embedding process for you with no custom code!

[Pyvespa: Building cost-efficient retrieval-augmented personal AI assistants](https://pyvespa.readthedocs.io/en/latest/examples/scaling-personal-ai-assistants-with-streaming-mode-cloud.html)
demonstrates how to use [Vespa streaming mode](https://docs.vespa.ai/en/streaming-search.html)
for cost-efficient retrieval for applications that store and retrieve personal data.
This notebook connects a custom [LlamaIndex Retriever](https://docs.llamaindex.ai/)
with a Vespa app using streaming mode to retrieve personal data.



## Visual search
[Pyvespa: Vespa 🤝 ColPali: Efficient Document Retrieval with Vision Language Models](https://pyvespa.readthedocs.io/en/latest/examples/colpali-document-retrieval-vision-language-models-cloud.html).
This notebook demonstrates how to represent [ColPali](https://huggingface.co/vidore/colpali) in Vespa.
ColPali is a powerful visual language model that can generate embeddings for images and text.
In this notebook, we will use ColPali to generate embeddings for images of PDF _pages_ and store them in Vespa.
We will also store the base64 encoded image of the PDF page and some meta data like title and url.
We will then demonstrate how to retrieve the PDF pages using the embeddings generated by ColPali.

[Pyvespa: Scaling ColPALI (VLM) Retrieval](https://pyvespa.readthedocs.io/en/latest/examples/simplified-retrieval-with-colpali-vlm_Vespa-cloud.html)
This notebook demonstrates how to represent [ColPali](https://huggingface.co/vidore/colpali) in Vespa and to scale to large collections.
Also see the [Scaling ColPali to billions of PDFs with Vespa](https://blog.vespa.ai/scaling-colpali-to-billions/) blog post.

[Pyvespa: ColPali Ranking Experiments on DocVQA](https://pyvespa.readthedocs.io/en/latest/examples/colpali-benchmark-vqa-vlm_Vespa-cloud.html).
This notebook demonstrates how to reproduce the ColPali results on [DocVQA](https://huggingface.co/datasets/vidore/docvqa_test_subsampled) with Vespa.
The dataset consists of PDF documents with questions and answers.
We demonstrate how we can binarize the patch embeddings and replace the float MaxSim scoring with a hamming based MaxSim
without much loss in ranking accuracy but with a significant speedup (close to 4x) and reducing the memory (and storage) requirements by 32x.

[Pyvespa: PDF-Retrieval using ColQWen2 (ColPali) with Vespa](https://pyvespa.readthedocs.io/en/latest/examples/pdf-retrieval-with-ColQwen2-vlm_Vespa-cloud.html).
This notebook is a continuation of our notebooks related to the ColPali models for complex document retrieval.
This notebook demonstrates using the new ColQWen2 model checkpoint.



## Ranking
[Pyvespa: Using Mixedbread.ai cross-encoder for reranking in Vespa.ai](https://pyvespa.readthedocs.io/en/latest/examples/cross-encoders-for-global-reranking.html).
With Vespa’s phased ranking capabilities,
doing cross-encoder inference for a subset of documents at a later stage in the ranking pipeline
can be a good trade-off between ranking performance and latency.
In this notebook, we show how to use the [Mixedbread.ai](https://www.mixedbread.ai/)
cross-encoder for [global-phase reranking](https://docs.vespa.ai/en/reference/schema-reference.html#using-a-global-phase-expression) in Vespa.

[Pyvespa: Standalone ColBERT with Vespa for end-to-end retrieval and ranking](https://pyvespa.readthedocs.io/en/latest/examples/colbert_standalone_Vespa-cloud.html).
This notebook illustrates using [ColBERT](https://github.com/stanford-futuredata/ColBERT) package to produce token vectors,
instead of using the native Vespa [ColBERT embedder](https://docs.vespa.ai/en/embedding.html#colbert-embedder).
This guide illustrates how to feed and query using a single passage representation:
* Compress token vectors using binarization compatible with Vespa's `unpack_bits` used in ranking.
  This implements the binarization of token-level vectors using `numpy`.
* Use Vespa [hex feed format](https://docs.vespa.ai/en/reference/document-json-format.html#tensor) for binary vectors.
* Query examples.

As a bonus, this also demonstrates how to use ColBERT end-to-end with Vespa for both retrieval and ranking.
The retrieval step searches the binary token-level representations using hamming distance.
This uses 32 nearestNeighbor operators in the same query, each finding 100 nearest hits in hamming space.
Then the results are re-ranked using the full-blown MaxSim calculation.

[Pyvespa: Standalone ColBERT + Vespa for long-context ranking](https://pyvespa.readthedocs.io/en/latest/examples/colbert_standalone_long_context_Vespa-cloud.html).
This is a guide on how to use the [ColBERT](https://github.com/stanford-futuredata/ColBERT) package to produce token-level vectors.
This as an alternative to using the native Vespa [ColBERT embedder](https://docs.vespa.ai/en/embedding.html#colbert-embedder).
This guide illustrates how to feed multiple passages per Vespa document (long-context):
* Compress token vectors using binarization compatible with Vespa's `unpack_bits`.
* Use Vespa hex feed format for binary vectors with mixed vespa tensors.
* How to query Vespa with the ColBERT query tensor representation.

The main goal of [Pyvespa: LightGBM: Training the model with Vespa features](https://pyvespa.readthedocs.io/en/latest/examples/lightgbm-with-categorical.html)
is to deploy and use a LightGBM model in a Vespa application.
The following tasks will be accomplished throughout the tutorial:
1. Train a LightGBM classification model with variable names supported by Vespa.
2. Create Vespa application package files and export then to an application folder.
3. Export the trained LightGBM model to the Vespa application folder.
4. Deploy the Vespa application using the application folder.
5. Feed data to the Vespa application.
6. Assert that the LightGBM predictions from the deployed model are correct.

The main goal of [Pyvespa: LightGBM: Mapping model features to Vespa features](https://pyvespa.readthedocs.io/en/latest/examples/lightgbm-with-categorical-mapping.html)
is to show how to deploy a LightGBM model with feature names that do not match Vespa feature names.
The following tasks will be accomplished throughout the tutorial:
1. Train a LightGBM classification model with generic feature names that will not be available in the Vespa application.
2. Create an application package and include a mapping from Vespa feature names to LightGBM model feature names.
3. Create Vespa application package files and export then to an application folder.
4. Export the trained LightGBM model to the Vespa application folder.
5. Deploy the Vespa application using the application folder.
6. Feed data to the Vespa application.
7. Assert that the LightGBM predictions from the deployed model are correct.



## Performance
[Pyvespa: Feeding performance](https://pyvespa.readthedocs.io/en/latest/examples/feed_performance.html)
This explorative notebook intends to shine some light on the different modes of feeding documents to Vespa.
We will look at these 4 different methods:
* Using `VespaSync`
* Using `VespaAsync`
* Using `feed_iterable()`
* Using [Vespa CLI](https://docs.vespa.ai/en/vespa-cli)

Try [Feeding to Vespa Cloud](https://pyvespa.readthedocs.io/en/latest/examples/feed_performance_cloud.html)
to test feeding using Cloud.



## More advanced sample applications


### News search and recommendation tutorial 
The [news](news/) sample application used in the [Vespa tutorial](https://docs.vespa.ai/en/tutorials/news-1-getting-started.html).
This application demonstrates basic search functionality.

It also demonstrates how to build a recommendation system
where the approximate nearest neighbor search in a shared user/item embedding space
is used to retrieve recommended content for a user.
This app also demonstrates using [parent-child](https://docs.vespa.ai/en/parent-child.html) 
relationships in Vespa.


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


### Search as you type and query suggestions 
The [incremental-search](incremental-search/) application demonstrates search-as-you-type functionality, where for each keystroke of the user, it retrieves matching documents. 
It also demonstrates search suggestions (query auto-completion).


### Vespa as ML inference server (model-inference)
The [model-inference](model-inference/) application demonstrates 
using Vespa as a stateless ML model inference server
where Vespa takes care of distributing ML models to multiple serving containers,
offering horizontal scaling and safe deployment.
Model versioning and feature processing pipeline.


### Vespa Documentation Search
[vespa-documentation-search](https://github.com/vespa-cloud/vespa-documentation-search)
is the search application that powers [search.vespa.ai](https://search.vespa.ai/) -
refer to this for GitHub Actions automation.
This sample app is a good start for [automated deployments](https://cloud.vespa.ai/en/automated-deployments),
as it has system, staging and production test examples.
It uses the [Document API](https://docs.vespa.ai/en/document-api-guide.html)
both for regular PUT operations but also for UPDATE with _create-if-nonexistent_.


### CORD-19 Search
[cord19.vespa.ai](https://cord19.vespa.ai/) is a full-featured application,
based on the [Covid-19 Open Research Dataset](https://huggingface.co/datasets/allenai/cord19):
* [cord-19](https://github.com/vespa-engine/cord-19): frontend
* [cord-19-search](https://github.com/vespa-cloud/cord-19-search): search backend


<!--
[travis](travis)
[part-purchases-demo](part-purchases-demo): A sample Vespa application to assist with learning how to group according to the [Grouping Guide](https://docs.vespa.ai/en/grouping.html).
[generic-request-processing](generic-request-processing)
http-api-using-*
-->

----

Note: Applications with _pom.xml_ are Java/Maven projects and must be built before deployment.
Refer to the [Developer Guide](https://docs.vespa.ai/en/developer-guide.html) for more information.

[Contribute](https://github.com/vespa-engine/vespa/blob/master/CONTRIBUTING.md) to the Vespa sample applications.

----

[![Vespa Sampleapps Search Feed](https://github.com/vespa-engine/sample-apps/actions/workflows/feed.yml/badge.svg)](https://github.com/vespa-engine/sample-apps/actions/workflows/feed.yml)

[![sample-apps link checker](https://api.screwdriver.cd/v4/pipelines/7038/link-checker-sample-apps/badge)](https://cd.screwdriver.cd/pipelines/7038/)

[![sample-apps build](https://api.screwdriver.cd/v4/pipelines/7038/build-apps/badge)](https://cd.screwdriver.cd/pipelines/7038/)

[![sample-apps verify-guides](https://api.screwdriver.cd/v4/pipelines/7038/verify-guides/badge)](https://cd.screwdriver.cd/pipelines/7038/)
[![sample-apps verify-guides-big](https://api.screwdriver.cd/v4/pipelines/7038/verify-guides-big/badge)](https://cd.screwdriver.cd/pipelines/7038/)
