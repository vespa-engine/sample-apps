# Vespa ranking experiments

Code to perform Vespa ranking experiments.

## Motivation

[nativeRank](https://docs.vespa.ai/en/reference/nativerank.html) is the current (Vespa 7) default sparse ranking 
for text. However, experiments ran against the [Beir benchmark](https://github.com/beir-cellar/beir) show that
BM25 outperforms nativeRank across the Beir datasets. Since BM25 requires more memory and disk space than nativeRank,
we decided to investigate if there was a cheaper combination of 
[Vespa ranking features](https://docs.vespa.ai/en/reference/rank-features.html) that would improve the 
ranking performance over nativeRank.

## Requirements

Install the required python libraries:

```shell
pip install -r requirements.txt
```

We assume that Docker is installed, as it is used to deploy the Vespa app used in the experiments. 

## Dataset

We need data to perform our experiments. We follow standard practice and use the 
[MS MARCO](https://microsoft.github.io/msmarco/) passage ranking dataset.

```python
from assessment import download_and_unzip_dataset

data_folder = download_and_unzip_dataset(data_dir="data", dataset_name="msmarco")
```

The code snipped above uses the `beir` library to download and unzip the dataset into the `data_folder`.

## Sampling data

The full msmarco dataset is very large, containing more than 8 million passages. This large size is unnecessary
when searching simple sparse ranking functions, meaning that we get reliable results with a subset of the data.

The following code snippet will load the full msmarco dataset we downloaded, and it will create a smaller dataset 
containing 1.000 train queries, 1.000 dev queries and 12.000 documents. Those 12.000 documents include 1.000 relevant 
documents associated with the sampled training queries and 1.000 relevant documents associated with the sampled
dev queries.

```python
from assessment import sample_data

sampled_data = sample_data(
    data_dir="data/msmarco",
    number_positive_samples=1000,
    number_negative_samples=10000,
    output_file="data/msmarco_sample.json",
)
```

In case you want to reuse a previously sampled dataset:

```python
from assessment import load_sampled_data

sampled_data = load_sampled_data(file="data/msmarco_sample.json")
```

## Create the application package

Create the application package containing a random ranking function that is used to collect training and
evaluation data. It also contains ranking functions that we want to evaluate, such as native rank and bm25 ranking
functions.

```python
from assessment import create_application_package

app_package = create_application_package()
```

## Deploy the Vespa app

Deploy the application package in a Docker container:

```python
from assessment import deploy_app_package

app = deploy_app_package(app_package=app_package, port=8080)
```

Establish a connection in case the application is already deployed:

```python
from vespa.application import Vespa

app = Vespa(url="http://localhost", port=8080)
```

## Feed data to the deployed app

Feed the sampled data corpus to the deployed app:

```python
from assessment import feed_app

feed_app(app=app, corpus=sampled_data["corpus"])
```

This only needs to be done once unless changes are made to the application that required a refeed.

## Collect training and validation data

We need to establish a ranking function to collect training and validation data. 

### Matching phase

Since we want to experiment with first-phase ranking functions, we need to use the same matching phase that will be used 
in production. Experiments run on the Beir benchmark shows that the 
[weakAnd](https://docs.vespa.ai/en/using-wand-with-vespa.html) operator performed on par with the more
expensive OR operator. So that is what we use as the matching phase in our experiments.

### Ranking phase

After running a series of experiments, we confirmed our initial hypotheses that randomly returning negative samples
out of the matched documents provided the best correlation between evaluation metrics computed based on sampled
data (offline) and evaluation metrics computed using the deployed app with the entire indexed corpus (online). 
Such a correlation is important because it is much cheaper to run offline experiments, but the results of those
experiments must agree with what we would observe when running the online system.

We therefore define a query model that uses weakAnd as the matching phase, and a random ranking function. For
each (query, relevant_doc)-pair we will return 99 non-relevant documents, totalling 100 documents per query. 

```python
from vespa.query import QueryModel, WeakAnd, RankProfile

number_additional_docs = 100
query_model = QueryModel(
    match_phase=WeakAnd(hits=number_additional_docs),
    rank_profile=RankProfile(name="random", list_features=True),
)
```

We use the query model defined above and the sampled data to collect training and dev datasets, which are stored
on the .csv files defined below:

```python
from assessment import collect_vespa_features

collect_vespa_features(
    app=app,
    query_model=query_model,
    number_additional_docs=number_additional_docs,
    train_qrels=sampled_data["train_qrels"],
    train_queries=sampled_data["train_queries"],
    dev_qrels=sampled_data["dev_qrels"],
    dev_queries=sampled_data["dev_queries"],
    train_output_file="msmarco_train_random_sample.csv",
    dev_output_file="msmarco_dev_random_sample.csv",
)
```

## TensorFlow ranking experiments

We want to optimize Normalized Discounted Cumulative Gain at position 10 (NDCG@10), which is the same metric used
for the Beir leaderboard. Previous experiments have shown that a listwise loss function that aims to optimize that
metric yields more correlated results to the ones obtained by a running Vespa instance.

That is the reason we picked [TensorFlow ranking](https://www.tensorflow.org/ranking). It contains the 
[ApproxNDCGLoss](https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/losses/ApproxNDCGLoss) loss function
and the [NDCGMetric](https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/metrics/NDCGMetric) evaluation 
metric used in our experiments reported in this section.

### Positive-constrained linear model search

In this experiment, we constrained the weights of the linear 
regression models to be greater than or equal to zero. We then 
eliminate one feature at a time, choosing to eliminate the 
feature with the smallest weight at each iteration.

Following is the complete feature list explored. We exclude `bm25(body)` because we want
to find a feature combination that improves on `nativeRank` without using bm25.

```python
feature_names = [
    "elementCompleteness(body).completeness",
    "elementCompleteness(body).fieldCompleteness",
    "elementCompleteness(body).queryCompleteness",
    "fieldMatch(body)",
    "fieldMatch(body).absoluteOccurrence",
    "fieldMatch(body).absoluteProximity",
    "fieldMatch(body).completeness",
    "fieldMatch(body).earliness",
    "fieldMatch(body).fieldCompleteness",
    "fieldMatch(body).gapLength",
    "fieldMatch(body).gaps",
    "fieldMatch(body).head",
    "fieldMatch(body).importance",
    "fieldMatch(body).longestSequence",
    "fieldMatch(body).longestSequenceRatio",
    "fieldMatch(body).matches",
    "fieldMatch(body).occurrence",
    "fieldMatch(body).orderness",
    "fieldMatch(body).outOfOrder",
    "fieldMatch(body).proximity",
    "fieldMatch(body).queryCompleteness",
    "fieldMatch(body).relatedness",
    "fieldMatch(body).segmentDistance",
    "fieldMatch(body).segmentProximity",
    "fieldMatch(body).segments",
    "fieldMatch(body).significance",
    "fieldMatch(body).significantOccurrence",
    "fieldMatch(body).tail",
    "fieldMatch(body).unweightedProximity",
    "fieldMatch(body).weight",
    "fieldMatch(body).weightedAbsoluteOccurrence",
    "fieldMatch(body).weightedOccurrence",
    "nativeFieldMatch",
    "nativeProximity",
    "nativeRank",
    "queryTermCount",
    "textSimilarity(body).fieldCoverage",
    "textSimilarity(body).order",
    "textSimilarity(body).proximity",
    "textSimilarity(body).queryCoverage",
    "textSimilarity(body).score",
]



```

The initial idea was to eliminate the features with weight equal to
zero one by one, but we noticed that the best result (given the model constrains) 
was obtained when leaving just one feature in the model, which was 
`fieldMatch(body).significance` for the run with the parameters below:

```python
from assessment import positive_constrained_linear_model_experiment

positive_constrained_linear_model_experiment(
    collected_train_file="msmarco_train_random_sample.csv",
    collected_dev_file="msmarco_dev_random_sample.csv",
    feature_names=feature_names,
    final_epoch=300,
    tune_early_stop_patience=10,
    tune_epochs=100,
    tuner_max_trials=10,
    tuner_dir="data/keras_tuner",
    tuner_project_name="positive_constrained_linear_experiment",
    output_file="positive_constrained_linear_weights.json",
)
```

The script will take training and dev data that we collected from the Vespa app and start by fitting a model
containing all the features. It will then remove one feature at a time and store both the feature names and 
evaluation metric for each iteration.

## Vespa evaluation

```python
from assessment import compare_offline_and_online_evaluation

compare_offline_and_online_evaluation(
    app=app,
    collected_train_file="msmarco_train_random_sample.csv",
    collected_dev_file="msmarco_dev_random_sample.csv",
    list_number_docs_per_query=[100],
    sampled_data=sampled_data,
    output_file="online_offline_comparison.csv",
)
```

We can see below that offline and online evaluation metrics reach similar conclusions about the quality of each
ranking function.

* Offline

| number_documents_per_query | bm25_offline_eval | significance_offline_eval | native_offline_eval | 
| -------------------------- | ----------------- | ------------------------- | ------------------- | 
| 100                        | 0.92              | 0.77                      | 0.70                |  

* Online

| number_documents_per_query | bm25_online_eval  | significance_online_eval  | native_online_eval  |
| -------------------------- | ----------------- | ------------------------- | ------------------- |
| 100                        |  0.79             | 0.63                      | 0.52                | 

