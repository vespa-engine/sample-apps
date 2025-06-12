<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# The RAG Blueprint

Start with this if you want to create a RAG application that
* delivers state-of-the-art quality
* with any amount of data, query load.

This requires at least Vespa 8.519.55.

This README provides the commands necessary to create, deploy, feed, and evaluate this RAG blueprint application.

For an in-depth tutorial with more reasoning and explanation, please see the [RAG Blueprint tutorial](TODO).

## Dataset

For this blueprint, we will use a synthetic dataset of documents belonging to a persona, Alex Chen, who is an AI Engineer at a fictional YC-backed startup called "SynapseFlow". The document dataset contains a mix of Alex's personal notes, technical documentation, workout logs, and other relevant information that reflects his professional and personal interests.

By feeding this dataset to Vespa, we enable a Retrieval-Augmented Generation (RAG) application to retrieve relevant documents and generate responses and insights from all Alex's documents. With Vespa, this could scale to billions of documents and thousands of queries per second, while still delivering state-of-the-art quality.

## Prerequisites

* [Docker](https://www.docker.com/) Desktop installed and running. 10GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* Minimum **8 GB** memory dedicated to Docker (the default is 2 GB on Macs)
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).

## Installing vespa-cli

This tutorial uses [Vespa-CLI](https://docs.vespa.ai/en/vespa-cli.html),
Vespa CLI is the official command-line client for Vespa.ai.
It is a single binary without any runtime dependencies and is available for Linux, macOS and Windows.

<pre>
$ brew install vespa-cli
</pre>

<pre data-test="exec">
$ vespa clone rag-blueprint rag-blueprint && cd rag-blueprint
</pre>

<pre data-test="exec">
$ docker run --detach --name vespa-rag --hostname vespa-rag \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19112:19112 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
</pre>

<pre data-test="exec">
$ vespa deploy --wait 300 ./app
</pre>

<pre data-test="exec">
$ vespa feed dataset/docs.jsonl
</pre>

<pre data-test="exec" data-test-assert-contains="yc_b2b_sales_workshop_notes.md">
$ vespa query 'query=yc b2b sales'
</pre>

## LLM-generation with OpenAI-client

The recommended way of providing an API key is through using the Secret Store in Vespa Cloud.
To enable this, you need to create a vault (if you don't have one already) and a secret through the Vespa Cloud console. If your vault is named `sample-apps` and contains a secret with the name `openai-api-key`, you would use the following configuration in your `services.xml` to set up the OpenAI client to use that secret:

```xml
  <secrets>
      <openai-api-key vault="sample-apps" name="openai-dev" />
  </secrets>
  <!-- Setup the client to OpenAI -->
  <component id="openai" class="ai.vespa.llm.clients.OpenAI">
      <config name="ai.vespa.llm.clients.llm-client">
          <apiKeySecretName>openai-api-key</apiKeySecretName>
      </config>
  </component>
```

Alternatively, for local deployments, you can set the `X-LLM-API-KEY` header in your query to use the OpenAI client for generation.

To test generation using the OpenAI client, post a query that runs the `openai` search chain, with `format=sse`. (Use `format=json` for a streaming json response including both the search hits and the LLM-generated tokens.)
<pre>
$ vespa query \
    --timeout 60 \
    --header="X-LLM-API-KEY:<your-api-key>" \
    yql='select *
    from doc
    where userInput(@query) or
    ({label:"title_label", targetHits:100}nearestNeighbor(title_embedding, embedding)) or
    ({label:"chunks_label", targetHits:100}nearestNeighbor(chunk_embeddings, embedding))' \
    query="Summarize the key architectural decisions documented for SynapseFlow's v0.2 release." \
    searchChain=openai \
    format=sse \
    hits=5 \
    traceLevel=1
</pre>

## Using a query profile

As an alternative to providing query parameters directly, Vespa supports [query-profiles](https://docs.vespa.ai/en/query-profiles.html?mode=selfhosted#using-a-query-profile), which allow you to define a set of query parameters to support different use cases. 
For this sample app, we have added a query profile named `rag`, see `app/search/query-profiles/rag.xml`.

<pre>
$ vespa query \
    --timeout 60 \
    --header="X-LLM-API-KEY:<your-api-key>" \
    query="Summarize the key architectural decisions documented for SynapseFlow's v0.2 release." \
    queryProfile=rag
</pre>

## Evaluating and improving ranking

### 1. Retrieval (match-phase) evals

We want to make sure we match all relevant docs.
We can do this easily using pyvespa TODO (wait for VespaMatchEvaluator PR). 

### 2. First-phase ranking

The goal of first-phase ranking is to create a good, but cheap proxy for final relevance score.
A linear combination of text and semantic features is a common starting point, which we will use for this blueprint.
Below, we show one way of finding this linear expression.

#### Collect rank features

In the rank-profile `collect-training-data`, you can see we have createad both text-matching features (bm25), semantic similarity (embedding closeness), as well as document-level and chunk-level features. These are not normalized to the same range, which mean that we should learn the relationship (coefficients) between them.
These will now be calculated and returned as part of the Vespa response when this rank-profile is used.

Too see an example of the response we can get from vespa, issue the command below (or inspect the included `resp.json`)
<pre>
vespa query \
    query="Summarize the key architectural decisions documented for SynapseFlow's v0.2 release." \
    queryProfile=hybrid \
    ranking.profile=collect-training-data > resp.json
</pre>

To do this for all our queries, we can run:

<pre>
python eval/collect_training_data.py
</pre>

This gives us a file with our defined feature values, and a binary relevance label for our relevant documents, as well as an equal number of random documents per query.

#### Learned linear model

To find the expression that best fits our dataset, we train a simple `LogisticRegression`-model, using stratified 5-fold cross-validation.

<pre>
python eval/train_logistic_regression.py
</pre>

which gives us this output:

```txt
Model Coefficients (trained on full data):
----------------------------------------
bm25(chunks)                  : -0.0172
bm25(title)                   : 0.5504
closeness(chunk_embeddings)   : -0.0005
closeness(title_embedding)    : -0.0029
max_chunk_sim_scores          : -0.0005
max_chunk_text_scores         : 0.7143
Intercept                     : -3.5974
----------------------------------------
```

We can translate this to our ranking expression, which we add to our `learned-linear` rank-profile:

```txt
-3.5974 -0.0029 * closeness(title_embedding) -
0.0005 * closeness(chunk_embeddings) +
0.5504 * bm25(title) -
0.0172 * bm25(chunks) -
0.0005 * max_chunk_sim_scores() +
0.7143 * max_chunk_text_scores()
```