<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa Product Ranking 

This sample application is used to demonstrate how to improve Product Search with Learning to Rank (LTR).

Blog post series:

* [Improving Product Search with Learning to Rank - part one](https://blog.vespa.ai/improving-product-search-with-ltr/)
This post introduces the dataset used in this sample application and several baseline ranking models. 
* [Improving Product Search with Learning to Rank - part two](https://blog.vespa.ai/improving-product-search-with-ltr-part-two/)
This post demonstrates how to train neural methods for search ranking. The neural training routine is found in this
[notebook](https://github.com/vespa-engine/sample-apps/blob/master/commerce-product-ranking/notebooks/train_neural.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vespa-engine/sample-apps/blob/master/commerce-product-ranking/notebooks/train_neural.ipynb).
* [Improving Product Search with Learning to Rank - part three](https://blog.vespa.ai/improving-product-search-with-ltr-part-three/)
This post demonstrates how to train GBDT methods for search ranking. The model uses also neural signals as features. See notebooks:
[XGBoost](https://github.com/vespa-engine/sample-apps/blob/master/commerce-product-ranking/notebooks/Train-xgboost.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vespa-engine/sample-apps/blob/master/commerce-product-ranking/notebooks/Train-xgboost.ipynb) and 
[LightGBM](https://github.com/vespa-engine/sample-apps/blob/master/commerce-product-ranking/notebooks/Train-lightgbm.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vespa-engine/sample-apps/blob/master/commerce-product-ranking/notebooks/Train-lightgbm.ipynb) 

This work uses the largest product relevance dataset released by Amazon:

> We introduce the “Shopping Queries Data Set”, a large dataset of difficult search queries,
> released with the aim of fostering research in the area of semantic matching of queries and products.
> For each query, the dataset provides a list of up to 40 potentially relevant results,
> together with ESCI relevance judgements (Exact, Substitute, Complement, Irrelevant)
> indicating the relevance of the product to the query.
> Each query-product pair is accompanied by additional information.
> The dataset is multilingual, as it contains queries in English, Japanese, and Spanish.

The dataset is found at [amazon-science/esci-data](https://github.com/amazon-science/esci-data). 
The dataset is released under the [Apache 2.0 license](https://github.com/amazon-science/esci-data/blob/main/LICENSE).

## Quick start

The following is a quick start recipe on how to get started with this application. 

* [Docker](https://www.docker.com/) Desktop installed and running. 6 GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download 
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* zstd: `brew install zstd`
* Python3 with `requests` `pyarrow` and `pandas` installed 

Validate Docker resource settings, should be minimum 6 GB:
<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Verify that configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application:
<pre data-test="exec">
$ vespa clone commerce-product-ranking my-app && cd my-app
</pre>

Download cross-encoder model:
<pre data-test="exec">
$ curl -L -o application/models/title_ranker.onnx \
    https://data.vespa-cloud.com/sample-apps-data/title_ranker.onnx
</pre>

See [scripts/export-bi-encoder.py](scripts/export-bi-encoder.py) and
[scripts/export-cross-encoder.py](scripts/export-cross-encoder.py) for how
to export models from PyTorch to ONNX format. 

Deploy the application:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 600 application
</pre>

If the above fails, check the logs:

<pre data-test="exec" >
$ docker logs vespa
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started#deploy-sample-applications).

## Run basic system test

This step is optional, but it indexes two 
documents and runs a query [test](https://docs.vespa.ai/en/reference/testing.html)

<pre data-test="exec" data-test-assert-contains="Success">
$ (cd application; vespa test tests/system-test/feed-and-search-test.json)
</pre>


## Indexing sample product data

Download the pre-processed sample product data for 16 products:

<pre data-test="exec">
$ zstdcat sample-data/sample-products.jsonl.zstd | vespa feed -
</pre>


## Evaluation 

Evaluate the `semantic-title` rank profile using the evaluation 
script ([scripts/evaluate.py](scripts/evaluate.py)).

Install requirements

<pre data-test="exec">
pip3 install numpy pandas pyarrow requests
</pre>

<pre data-test="exec">
$ python3 scripts/evaluate.py \
  --endpoint http://localhost:8080/search/ \
  --example_file sample-data/test-sample.parquet \
  --ranking semantic-title 
</pre>

[evaluate.py](scripts/evaluate.py) runs all the queries in the test split using the `--ranking` `<rank-profile>`
and produces a `<ranking>.run` file with the top ranked results.
This file is formatted in the format that `trec_eval` expects.

<pre data-test="exec" data-test-assert-contains="B08PB9TTKT">
$ cat semantic-title.run 
</pre>

Example ranking produced by Vespa using the `semantic-title` rank-profile for query 535:
<pre>
535 Q0 B08PB9TTKT 1 0.46388297538130346 semantic-title
535 Q0 B00B4PJC9K 2 0.4314163871097326 semantic-title
535 Q0 B0051GN8JI 3 0.4199624989861286 semantic-title
535 Q0 B084TV3C1B 4 0.4177780086570998 semantic-title
535 Q0 B08NVQ8MZX 5 0.4175260475587483 semantic-title
535 Q0 B00DHUA9VA 6 0.41558328517364673 semantic-title
535 Q0 B08SHMLP5S 7 0.41512211873088406 semantic-title
535 Q0 B08VSJGP1N 8 0.41479904241634674 semantic-title
535 Q0 B08QGZMCYQ 9 0.41107229418202607 semantic-title
535 Q0 B0007KPRIS 10 0.4073851390694049 semantic-title
535 Q0 B08VJ66CNL 11 0.4040355668337184 semantic-title
535 Q0 B000J1HDWI 12 0.40354871020728317 semantic-title
535 Q0 B0007KPS3C 13 0.39775755175088207 semantic-title
535 Q0 B0072LFB68 14 0.39334250744409155 semantic-title
535 Q0 B01M0SFMIH 15 0.3920197770681833 semantic-title
535 Q0 B0742BZXC2 16 0.3778094352830984 semantic-title
</pre>

This run file can then be evaluated using the [trec_eval](https://github.com/usnistgov/trec_eval) utility.

Download a pre-processed query-product relevance judgments in TREC format:
<pre data-test="exec">
$  curl -L -o test.qrels \
    https://data.vespa-cloud.com/sample-apps-data/test.qrels
</pre>

Install `trec_eval` (your mileage may vary):

<pre data-test="exec">
git clone --depth 1 --branch v9.0.8 https://github.com/usnistgov/trec_eval && cd trec_eval && make install && cd ..
</pre>

Run evaluation :

<pre data-test="exec" data-test-assert-contains="all">
$ trec_eval test.qrels semantic-title.run -m 'ndcg.1=0,2=0.01,3=0.1,4=1'
</pre>

This particular product ranking for the query produces a NDCG score of 0.7046. 
Note that the `sample-data/test-sample.parquet` file only contains one query.
To get the overall score, one must compute all the NDCG scores of all queries in the
test split and report the *average* NDCG score.  

Note that the evaluation uses custom NDCG label gains:

- Label 1 is **I**rrelevant with 0 gain
- Label 2 is **S**upplement with 0.01 gain
- Label 3 is **C**omplement with 0.1 gain
- Label 4 is **E**xact with 1 gain

We can also try another ranking model:

<pre data-test="exec">
$ python3 scripts/evaluate.py \
  --endpoint http://localhost:8080/search/ \
  --example_file sample-data/test-sample.parquet \
  --ranking cross-title
</pre>

<pre data-test="exec" data-test-assert-contains="all">
$ trec_eval test.qrels cross-title.run -m 'ndcg.1=0,2=0.01,3=0.1,4=1'
</pre>

Which for this query produces a NDCG score of 0.8208, better than the semantic-title model.


## Shutdown and remove the Docker container

<pre data-test="after">
$ docker rm -f vespa
</pre>


## Full evaluation 

Download a pre-processed feed file with all (1,215,854) products:

<pre>
$  curl -L -o product-search-products.jsonl.zstd \
    https://data.vespa-cloud.com/sample-apps-data/product-search-products.jsonl.zstd
</pre>

This step is resource intensive as the semantic embedding model encodes 
the product title and description into the dense embedding vector space.

<pre>
$ zstdcat product-search-products.jsonl.zstd | vespa feed -
</pre>

Evaluate the `hybrid` baseline rank profile using the evaluation 
script ([scripts/evaluate.py](scripts/evaluate.py)).

<pre>
$ python3 scripts/evaluate.py \
  --endpoint http://localhost:8080/search/ \
  --example_file "https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet?raw=true" \
  --ranking semantic-title 
</pre>

For Vespa cloud deployments we need to pass certificate and the private key.

<pre>
$ python3 scripts/evaluate.py \
  --endpoint https://productsearch.samples.aws-us-east-1c.perf.z.vespa-app.cloud/search/ \
  --example_file "https://github.com/amazon-science/esci-data/blob/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet?raw=true" \
  --ranking semantic-title \
  --cert &lt;path-to-data-plane-cert.pem&gt; \
  --key &lt;path-to-data-plane-private-key.pem&gt;
</pre>

Run evaluation using `trec_eval`:
<pre>
$ trec_eval test.qrels semantic-title.run -m 'ndcg.1=0,2=0.01,3=0.1,4=1
</pre>
