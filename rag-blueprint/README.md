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
* Python 3.8 or later. We recommend using [uv](https://docs.astral.sh/uv/) to manage virtual environment and install python dependencies.

## Quick start

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

For local deployment using docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

<pre data-test="exec">
$ vespa deploy --wait 300 ./app
</pre>

<pre data-test="exec">
$ vespa feed dataset/docs.jsonl
</pre>

<pre data-test="exec" data-test-assert-contains="yc_b2b_sales_workshop_notes.md">
$ vespa query 'query=yc b2b sales' presentation.summary="no-chunks"
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
    hits=5
</pre>

## Using query profiles for different use cases

As an alternative to providing query parameters directly, Vespa supports [query-profiles](https://docs.vespa.ai/en/query-profiles.html?mode=selfhosted#using-a-query-profile), which allow you to define a set of query parameters to support different use cases.
For this sample app, we have added 3 query profiles:

1. `rag`, see `app/search/query-profiles/rag.xml`.
2. `hybrid`, see `app/search/query-profiles/hybrid.xml`.
3. `deepresearch`, see `app/search/query-profiles/deepresearch.xml`.

Which all have different query parameters set, such as the search chain to use, the ranking profile, and the number of hits to return.
The command below will use that query profile to set parameters listed in previous section.

### `rag` query profile

Run the command below to use the `rag` query profile.

<pre>
$ vespa query \
    --header="X-LLM-API-KEY:<your-api-key>" \
    query="Summarize the key architectural decisions documented for SynapseFlow's v0.2 release." \
    queryProfile=rag
</pre>

### `hybrid` query profile

Run the command below to use the `hybrid` query profile.

<pre>
$ vespa query \
    query="Summarize the key architectural decisions documented for SynapseFlow's v0.2 release." \
    queryProfile=hybrid
</pre>

### `deepresearch` query profile

Run the command below to use the `deepresearch` query profile.
<pre>
$ vespa query \
    query="Summarize the key architectural decisions documented for SynapseFlow's v0.2 release." \
    queryProfile=deepresearch
</pre>

## Evaluating and improving ranking

### 1. Retrieval (match-phase) evals

We want to make sure we match all relevant docs.
We can do this quite easily using pyvespa's [VespaMatchEvaluator](https://vespa-engine.github.io/pyvespa/api/vespa/evaluation.html#vespa.evaluation.VespaMatchEvaluator).

We defined 3 different YQL queries that we will evaluate separately:

1. **`semantic`**: This query uses the `nearestNeighbor` operator to find documents based on semantic similarity.

2. **`weakand`**: This query uses the `userQuery` operator to find documents based on text matching with weak AND semantics.

3. **`hybrid`**: This query uses a combination of text and semantic matching to find documents.

To run the match-phase evaluation for all of them, run the command below.

<pre>
python eval/evaluate_match_phase.py
</pre>

And expect the following output:

#### Semantic Query Evaluation

```sql
select * from doc where 
({targetHits:100}nearestNeighbor(title_embedding, embedding)) or
({targetHits:100}nearestNeighbor(chunk_embeddings, embedding))
```

| Metric                    | Value    |
| ------------------------- | -------- |
| Match Recall              | 1.0000   |
| Average Recall per Query  | 1.0000   |
| Total Relevant Documents  | 51       |
| Total Matched Relevant    | 51       |
| Average Matched per Query | 100.0000 |
| Total Queries             | 20       |
| Search Time Average (s)   | 0.0090   |
| Search Time Q50 (s)       | 0.0060   |
| Search Time Q90 (s)       | 0.0193   |
| Search Time Q95 (s)       | 0.0220   |

#### WeakAnd Query Evaluation

```sql
select * from doc where userQuery()
```

| Metric                    | Value   |
| ------------------------- | ------- |
| Match Recall              | 1.0000  |
| Average Recall per Query  | 1.0000  |
| Total Relevant Documents  | 51      |
| Total Matched Relevant    | 51      |
| Average Matched per Query | 88.7000 |
| Total Queries             | 20      |
| Search Time Average (s)   | 0.0071  |
| Search Time Q50 (s)       | 0.0060  |
| Search Time Q90 (s)       | 0.0132  |
| Search Time Q95 (s)       | 0.0171  |

#### Hybrid Query Evaluation

```sql
select * from doc where 
({targetHits:100}nearestNeighbor(title_embedding, embedding)) or
({targetHits:100}nearestNeighbor(chunk_embeddings, embedding)) or
userQuery()
```

| Metric                    | Value    |
| ------------------------- | -------- |
| Match Recall              | 1.0000   |
| Average Recall per Query  | 1.0000   |
| Total Relevant Documents  | 51       |
| Total Matched Relevant    | 51       |
| Average Matched per Query | 100.0000 |
| Total Queries             | 20       |
| Search Time Average (s)   | 0.0076   |
| Search Time Q50 (s)       | 0.0055   |
| Search Time Q90 (s)       | 0.0150   |
| Search Time Q95 (s)       | 0.0201   |

### Conclusion

We can see that all queries match all relevant documents, which is expected, since we use `targetHits:100` in the `nearestNeighbor` operator, and this is also the default for `weakAnd`(and `userQuery`).

For a larger scale dataset, we could tune these parameters to find a good balance between recall and performance.

### 2. First-phase ranking

With our match-phase evaluation done, we can move on to the ranking phase.
We will start by collecting some training data for a handpicked set of features, which we will combine into a (cheap) linear first-phase ranking expression.

### Collect matchfeatures

In the rank-profile [`collect-training-data`](app/schemas/doc/collect-training-data.profile), you can see we have created both text-matching features (bm25), semantic similarity (embedding closeness), as well as document-level and chunk-level features. These are not normalized to the same range, which mean that we should learn the relationship (coefficients) between them.
These will now be calculated and returned as part of the Vespa response when this rank-profile is used.

We want to collect features from both the relevant documents, as well as a set of random documents (we sample an equal ratio of random and relevant documents), to ensure we have a good distribution of feature values.

To do this for all our queries, we can run:

<pre>
python eval/collect_pyvespa.py --collect_matchfeatures --collector_name matchfeatures-firstphase
</pre>

This will collect the 8 features defined in the `collect-training-data` rank-profile, and save them to a file to use as input for training our linear model.

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

We can translate this to our ranking expression, which we add to our `hybrid`  query-profile. We could add them directly to our `learned-linear` rank-profile, but by putting the coefficients in the query-profile, we can override them without having to redeploy the application.

Now, let us evaluate the performance of this first-phase ranking expression.

TODO: Add section about evaluating first-phase ranking.

### 3. Second-phase ranking

For the second-phase ranking, we can afford to use a more expensive ranking expression, since we will only run it on the top-k documents from the first-phase ranking (defined by rerank-count parameter).

For this, we will request Vespa's default set of rankfeatures, which includes a large set of text features, see [docs](https://docs.vespa.ai/en/reference/rank-features.html) for details.

To do this, we can run the same script as before, but with the added `--collect_rankfeatures` flag.

<pre>
python eval/collect_pyvespa.py --collect_rankfeatures --collect_matchfeatures --collector_name rankfeatures-secondphase
</pre>

We can see that we collected 196 features. Let us now train a GBDT model to predict the relevance_label (probability between 0 and 1) for each document, using the features we collected.
We use 5-fold cross-validation and set hyperparameters to prevent growing too large and deep trees, since we only have a small dataset, to avoid overfitting.

<pre>
python eval/train_lightgbm.py --input_file eval/output/Vespa-training-data_second-phase_20250619_182246.csv
</pre>

And you will get output like:

```txt
2025-06-20 05:34:53,991 - INFO - Loaded 102 rows × 200 columns
2025-06-20 05:34:53,995 - INFO - Dropping 116 constant columns
2025-06-20 05:34:53,995 - INFO - Dropping ID columns: ['query_id', 'doc_id', 'relevance_score']
2025-06-20 05:34:53,998 - INFO - Performing 5-Fold Stratified Cross-Validation...
2025-06-20 05:34:53,999 - INFO - Training Fold 1/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[38]    train's auc: 0.979268   valid's auc: 0.9
2025-06-20 05:34:54,055 - INFO - Fold 1: AUC = 0.9000, ACC = 0.7143
2025-06-20 05:34:54,056 - INFO - Training Fold 2/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[22]    train's auc: 0.991463   valid's auc: 0.945455
2025-06-20 05:34:54,100 - INFO - Fold 2: AUC = 0.9455, ACC = 0.9048
2025-06-20 05:34:54,100 - INFO - Training Fold 3/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[20]    train's auc: 0.991071   valid's auc: 0.97
2025-06-20 05:34:54,142 - INFO - Fold 3: AUC = 0.9700, ACC = 0.8500
2025-06-20 05:34:54,143 - INFO - Training Fold 4/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[24]    train's auc: 0.996429   valid's auc: 0.93
2025-06-20 05:34:54,185 - INFO - Fold 4: AUC = 0.9300, ACC = 0.8500
2025-06-20 05:34:54,185 - INFO - Training Fold 5/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[2]     train's auc: 0.928869   valid's auc: 1
2025-06-20 05:34:54,220 - INFO - Fold 5: AUC = 1.0000, ACC = 0.9000

------------------------------------------------------------
             Cross-Validation Results (5-Fold)             
------------------------------------------------------------
Metric             | Mean               | Std Dev           
------------------------------------------------------------
Accuracy           | 0.8438             | 0.0689            
ROC AUC            | 0.9491             | 0.0341            
------------------------------------------------------------
Overall CV AUC: 0.8965 • ACC: 0.8431
------------------------------------------------------------
2025-06-20 05:34:54,224 - INFO - Feature importance saved to Vespa-training-data_second-phase_20250619_182246_feature_importance.csv
2025-06-20 05:34:54,224 - INFO - Mean feature importance (gain):
2025-06-20 05:34:54,224 - INFO -   nativeProximity: 102.5542
2025-06-20 05:34:54,224 - INFO -   closeness(chunk_embeddings): 91.7256
2025-06-20 05:34:54,225 - INFO -   avg_top_3_chunk_sim_scores: 66.7286
2025-06-20 05:34:54,225 - INFO -   max_chunk_text_scores: 43.7979
2025-06-20 05:34:54,225 - INFO -   firstPhase: 37.8586
2025-06-20 05:34:54,225 - INFO -   nativeFieldMatch: 20.9317
2025-06-20 05:34:54,225 - INFO -   avg_top_3_chunk_text_scores: 20.6642
2025-06-20 05:34:54,225 - INFO -   nativeRank: 18.3844
2025-06-20 05:34:54,225 - INFO -   max_chunk_sim_scores: 15.1054
2025-06-20 05:34:54,225 - INFO -   elementCompleteness(chunks).completeness: 7.0140
2025-06-20 05:34:54,225 - INFO - Selected 16 features with non-zero importance
2025-06-20 05:34:54,225 - INFO - Training final model on all data for 21 rounds
2025-06-20 05:34:54,243 - INFO - Model exported to /Users/thomas/Repos/sample-apps/rag-blueprint/eval/Vespa-training-data_second-phase_20250619_182246_lightgbm_model.json
2025-06-20 05:34:54,243 - INFO - Training completed successfully!
```

We can see that for this small dataset, our most important features are the `nativeProximity`, `closeness(chunk_embeddings)`, `avg_top_3_chunk_sim_scores`, and `max_chunk_text_scores`. 


Great! We now have a trained GBDT model that we will use for our second-phase ranking.
To control the number of documents that will be exposed to second-phase, we can set the `rerank-count` parameter (default is 100).

We create a new rank-profile called `second-with-gbdt`, which uses the GBDT model we trained, and can use the `hybrid` query-profile, but override the ranking profile to use `second-with-gbdt` to test it out.

<pre>
vespa query query="what are key points learned for finetuning llms?" queryProfile=hybrid ranking=second-with-gbdt
</pre>

And of course, we can use the `rag` query profile to add the LLM generation of the response. 

<pre>
$ vespa query \
 --timeout 60 \
 --header="X-LLM-API-KEY:<your-api-key>" \
 query="what are key points learned for finetuning llms?" \
 queryProfile=rag \
 ranking=second-with-gbdt
</pre>

TODO: Add section about evaluating second-phase ranking.

Congratulations! You have now created a RAG application that can scale to billions of documents and thousands of queries per second, while still delivering state-of-the-art quality.
What will you build?
