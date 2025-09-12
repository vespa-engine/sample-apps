<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

# Evaluating and improving relevance

Your RAG solution won't be better than the quality you are able to surface for each problem
your LLM need to solve. That is why relevance improvements are most of the work of developing
a seriously good RAG application (once infrastructure and scaling is handled by Vespa).

This document provides a hands-on guide to evaluating and improving relevance in your Vespa application.

## Prerequisites

We'll use Python in this section, you'll need Python 3.8 or later.

Install uv to manage virtual environments and install python dependencies:
<pre>
$ curl -LsSf https://astral.sh/uv/install.sh | sh
</pre>

Activate the uv environment and install dependencies:

<pre>
$ cd eval && uv sync && cd ..
</pre>

## About the blueprint dataset

For this blueprint, we will use a synthetic dataset of documents belonging to a persona, Alex Chen,
who is an AI Engineer at a fictional YC-backed startup called "SynapseFlow". The document dataset
contains a mix of Alex's personal notes, technical documentation, workout logs, and other relevant
information that reflects his professional and personal interests.

To allow for quick iterations, and facilitate easier learning curve, we have restricted the dataset to
100 documents of varying length.

By feeding this dataset to Vespa, we enable a Retrieval-Augmented Generation (RAG) application to
retrieve relevant documents and generate responses and insights from all Alex's documents.
With Vespa, this can scale to billions of documents and thousands of queries per second,
while still delivering state-of-the-art quality.

## Evaluating and improving ranking

Now, we will show you how the query and rank profiles provided in the blueprint app were developed, 
using an evaluation-driven approach.

### 1. Retrieval (match-phase)

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

### Retrieval summary

We can see that all queries match all relevant documents, which is expected, since we use `targetHits:100`
in the `nearestNeighbor` operator, and this is also the default for `weakAnd`(and `userQuery`).

For a larger scale dataset, we could tune these parameters to find a good balance between recall and performance.

### 2. First-phase ranking

With our match-phase evaluation done, we can move on to the ranking phase.
We will start by collecting some training data for a handpicked set of features, which we will combine
into a (cheap) linear first-phase ranking expression.

### Collect matchfeatures

For this, we will use the [`collect-training-data`](app/schemas/doc/collect-training-data.profile). This profile inherits the [`base-features`](app/schemas/doc/base-features.profile),
where you can see we have created both text-matching features (bm25), semantic similarity (embedding closeness),
as well as document-level and chunk-level features. These are not normalized to the same range, which mean that
we should learn the relationship (coefficients) between them.
These will now be calculated and returned as part of the Vespa response when this rank-profile is used.

```txt
bm25(title)
bm25(chunks)
max_chunk_sim_scores
max_chunk_text_scores
avg_top_3_chunk_sim_scores
avg_top_3_chunk_text_scores
```

We want to collect features from both the relevant documents, as well as a set of random documents
(we sample an equal ratio of random and relevant documents), to ensure we have a good distribution of feature values.

To do this for all our queries, we can run:

<pre>
python eval/collect_pyvespa.py --collect_matchfeatures
</pre>

This will collect these 6 features defined in the inherited `base-features` rank-profile, and save them to a file
to use as input for training our linear model.

```txt
bm25(title)
bm25(chunks)
max_chunk_sim_scores
max_chunk_text_scores
avg_top_3_chunk_sim_scores
avg_top_3_chunk_text_scores
```

This gives us a file with our defined feature values, and a binary relevance label for our relevant documents,
as well as an equal number of random documents per query.

### Learned linear model

To find the expression that best fits our dataset, we train a simple `LogisticRegression`-model,
using stratified 5-fold cross-validation.

Note that we need to scale the features to have a mean of 0 and standard deviation of 1 before training,
and apply inverse scaling to the coefficients after training, in order to use the raw values from Vespa directly.

<pre>
python eval/train_logistic_regression.py
</pre>

which gives us this output:

```txt
Transformed Coefficients (for original unscaled features):
--------------------------------------------------
avg_top_3_chunk_sim_scores   : 13.383840
avg_top_3_chunk_text_scores  : 0.203145
bm25(chunks)                 : 0.159914
bm25(title)                  : 0.191867
max_chunk_sim_scores         : 10.067169
max_chunk_text_scores        : 0.153392
Intercept                    : -7.798639
--------------------------------------------------
```

We can translate this to our ranking expression, which we add to our `hybrid`  query-profile.
We could add them directly to our `learned-linear` rank-profile, but by putting the coefficients
in the query-profile, we can override them without having to redeploy the application.

Now, let us evaluate the performance of this first-phase ranking expression.

### Evaluate first-phase ranking

By running the following command

<pre>
python evaluate_ranking.py
</pre>

We run the evaluation script on a set of unseen test queries, and get the following output:

```json
{
    "accuracy@1": 1.0,
    "accuracy@3": 1.0,
    "accuracy@5": 1.0,
    "accuracy@10": 1.0,
    "precision@10": 0.235,
    "recall@10": 0.9405,
    "precision@20": 0.13,
    "recall@20": 0.9955,
    "mrr@10": 1.0,
    "ndcg@10": 0.8902,
    "map@100": 0.8197,
    "searchtime_avg": 0.017,
    "searchtime_q50": 0.0165,
    "searchtime_q90": 0.0251,
    "searchtime_q95": 0.0267
}
```

We can see that our results are already very good. This is of course due to the fact that we have a small,
synthetic dataset. In reality, you should align the metric expectations with your dataset and test queries.

For the first phase ranking, we care most about recall, as we just want to make sure our candidate documents
are ranked high enough to be included in the second-phase ranking. (the default number of documents that will be exposed to second-phase is 10 000, but can be controlled by the `rerank-count` parameter).

We can also see that our search time is quite fast, with an average of 17ms. You should consider whether
this is well within your latency budget, as you want some headroom for second-phase ranking.

### 3. Second-phase ranking

For the second-phase ranking, we can afford to use a more expensive ranking expression, since we will only
run it on the top-k documents from the first-phase ranking (defined by rerank-count parameter).

For this, we will request Vespa`s default set of rankfeatures, which includes a large set of text features,
see [docs](https://docs.vespa.ai/en/reference/rank-features.html) for details.

To do this, we can run the same script as before, but with the added `--collect_rankfeatures` flag.

<pre>
python eval/collect_pyvespa.py --collect_rankfeatures --collect_matchfeatures --collector_name rankfeatures-secondphase
</pre>

We can see that we collected 194 features. Let us now train a GBDT model to predict the relevance_label
(probability between 0 and 1) for each document, using the features we collected.
We use 5-fold cross-validation and set hyperparameters to prevent growing too large and deep trees,
since we only have a small dataset, to avoid overfitting.

For final training, we exclude features that have zero importance during the cross-validation,
and train the final model on all queries (not test queries).

<pre>
python eval/train_lightgbm.py --input_file eval/output/Vespa-training-data_match_rank_second_phase_20250623_135819.csv
</pre>

And you will get output like:

```txt
2025-06-23 14:02:35,681 - INFO - Loaded 102 rows × 198 columns
2025-06-23 14:02:35,686 - INFO - Dropping 116 constant columns
2025-06-23 14:02:35,686 - INFO - Dropping ID columns: ['query_id', 'doc_id', 'relevance_score']
2025-06-23 14:02:35,689 - INFO - Performing 5-Fold Stratified Cross-Validation...
2025-06-23 14:02:35,691 - INFO - Training Fold 1/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[43]    train's auc: 0.999695   valid's auc: 0.981818
2025-06-23 14:02:35,751 - INFO - Fold 1: AUC = 0.9818, ACC = 0.8571
2025-06-23 14:02:35,752 - INFO - Training Fold 2/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[1]     train's auc: 0.923077   valid's auc: 1
2025-06-23 14:02:35,782 - INFO - Fold 2: AUC = 1.0000, ACC = 1.0000
2025-06-23 14:02:35,783 - INFO - Training Fold 3/5
Training until validation scores don't improve for 50 rounds
[100]   train's auc: 1  valid's auc: 0.949495
Early stopping, best iteration is:
[92]    train's auc: 1  valid's auc: 0.949495
2025-06-23 14:02:35,857 - INFO - Fold 3: AUC = 0.9495, ACC = 0.8500
2025-06-23 14:02:35,857 - INFO - Training Fold 4/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[14]    train's auc: 0.994335   valid's auc: 1
2025-06-23 14:02:35,893 - INFO - Fold 4: AUC = 1.0000, ACC = 1.0000
2025-06-23 14:02:35,893 - INFO - Training Fold 5/5
Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[25]    train's auc: 0.997615   valid's auc: 1
2025-06-23 14:02:35,936 - INFO - Fold 5: AUC = 1.0000, ACC = 0.9000

------------------------------------------------------------
             Cross-Validation Results (5-Fold)             
------------------------------------------------------------
Metric             | Mean               | Std Dev           
------------------------------------------------------------
Accuracy           | 0.9214             | 0.0664            
ROC AUC            | 0.9863             | 0.0197            
------------------------------------------------------------
Overall CV AUC: 0.9249 • ACC: 0.9216
------------------------------------------------------------
2025-06-23 14:02:35,941 - INFO - Feature importance saved to Vespa-training-data_match_rank_second_phase_20250623_135819_feature_importance.csv
2025-06-23 14:02:35,941 - INFO - Mean feature importance (gain):
2025-06-23 14:02:35,942 - INFO -   nativeProximity: 168.8498
2025-06-23 14:02:35,942 - INFO -   firstPhase: 151.7382
2025-06-23 14:02:35,942 - INFO -   max_chunk_sim_scores: 69.4377
2025-06-23 14:02:35,942 - INFO -   avg_top_3_chunk_text_scores: 56.5079
2025-06-23 14:02:35,942 - INFO -   avg_top_3_chunk_sim_scores: 31.8700
2025-06-23 14:02:35,942 - INFO -   nativeRank: 20.0716
2025-06-23 14:02:35,942 - INFO -   nativeFieldMatch: 15.9914
2025-06-23 14:02:35,942 - INFO -   elementSimilarity(chunks): 9.7003
2025-06-23 14:02:35,942 - INFO -   bm25(chunks): 3.8777
2025-06-23 14:02:35,942 - INFO -   max_chunk_text_scores: 3.6406
2025-06-23 14:02:35,942 - INFO -   fieldTermMatch(chunks,4).firstPosition: 1.2615
2025-06-23 14:02:35,942 - INFO -   fieldTermMatch(chunks,4).occurrences: 1.0543
2025-06-23 14:02:35,942 - INFO -   fieldTermMatch(chunks,4).weight: 0.7264
2025-06-23 14:02:35,942 - INFO -   term(3).significance: 0.5078
2025-06-23 14:02:35,942 - INFO - Selected 14 features with non-zero importance
2025-06-23 14:02:35,942 - INFO - Training final model on all data for 35 rounds
2025-06-23 14:02:35,966 - INFO - Model exported to /Users/thomas/Repos/sample-apps/rag-blueprint/eval/Vespa-training-data_match_rank_second_phase_20250623_135819_lightgbm_model.json
2025-06-23 14:02:35,966 - INFO - Training completed successfully!
```

We can see that for this small dataset, our most important features are:

| Feature                                | Importance |
| -------------------------------------- | ---------- |
| nativeProximity                        | 168.8498   |
| firstPhase                             | 151.7382   |
| max_chunk_sim_scores                   | 69.4377    |
| avg_top_3_chunk_text_scores            | 56.5079    |
| avg_top_3_chunk_sim_scores             | 31.8700    |
| nativeRank                             | 20.0716    |
| nativeFieldMatch                       | 15.9914    |
| elementSimilarity(chunks)              | 9.7003     |
| bm25(chunks)                           | 3.8777     |
| max_chunk_text_scores                  | 3.6406     |
| fieldTermMatch(chunks,4).firstPosition | 1.2615     |
| fieldTermMatch(chunks,4).occurrences   | 1.0543     |
| fieldTermMatch(chunks,4).weight        | 0.7264     |
| term(3).significance                   | 0.5078     |

We can see that several of the more expensive text features has high importance. It is also reassuring to see
that the `firstPhase` feature, which is the output of our first-phase ranking, has a high importance,
meaning that it is a not too bad predictor of relevance for the second-phase ranking by itself.

We add the newly trained and exported lightgbm model to our Vespa application, and create a new
rank-profile called `second-with-gbdt` that will use this model.

Let us see how our performance on the unseen test queries looks like now.

### Evaluate second-phase ranking

By running our evaluation script with the `--second_phase` flag, we can evaluate the second-phase ranking
on the unseen test queries, using the `second-with-gbdt`-rank profile, containing the GBDT model we just trained.

<pre>
python evaluate_ranking.py --second_phase
</pre>

And the results we get are:

```json
{
    "accuracy@1": 0.9,
    "accuracy@3": 1.0,
    "accuracy@5": 1.0,
    "accuracy@10": 1.0,
    "precision@10": 0.23500000000000001,
    "recall@10": 0.9401515151515152,
    "precision@20": 0.12999999999999998,
    "recall@20": 0.9954545454545455,
    "mrr@10": 0.95,
    "ndcg@10": 0.8782459504293774,
    "map@100": 0.8091120429278325,
    "searchtime_avg": 0.020400000000000005,
    "searchtime_q50": 0.018000000000000002,
    "searchtime_q90": 0.0333,
    "searchtime_q95": 0.03615000000000001
}
```

We were not able to improve much on the already very good first phase ranking, but you would expect
significant improvements on a large real-world dataset.

Lets add a new query-profile that will inherit the previous `hybrid` query-profile, but will override
the ranking profile to use the `second-with-gbdt` rank-profile, and set the default number of hits to 20,
which (if our test queries are representative) should give us a recall of 0.99 for the second-phase ranking.

<pre>
vespa query query="what are key points learned for finetuning llms?" queryProfile=hybrid-with-gbdt
</pre>

And of course, we can also add a new `rag-with-gbdt`, that uses our new query profile, but overrides with
parameters to add the LLM generation of the response.

<pre>
$ vespa query \
 --timeout 60 \
 --header="X-LLM-API-KEY:<your-api-key>" \
 query="what are key points learned for finetuning llms?" \
 queryProfile=rag-with-gbdt
</pre>


### Further improvements

Finally, we will sketch out some opportunities for further improvements.
As you have seen, we started out with only binary relevance labels for a few queries, and trained a model
based on the relevant docs and a set of random documents.

This was useful initially, as we had no better way to retrieve the candidate documents.
Now, that we have a reasonably good second-phase ranking, we could potentially generate a new set of
relevance labels for queries that we did not have labels for by having an LLM do relevance judgments
of the top k returned hits. This training dataset would likely be even better in separating the top documents.

We also have the `global-phase` ranking, which could be suitable for doing a reranking of the top documents
from the second-phase ranking. Common options for global-phase are cross-encoders or another GBDT model,
trained for better separating top ranked documents. For RAG applications, we consider this less important
than for search applications where the results are mainly consumed by an human, as LLMs don't care that much
about the ordering of the results.

