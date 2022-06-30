<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# MS Marco Document Ranking
The baseline model for MS Marco *Document* Ranking using sparse lexical matching and a GBDT re-ranking model
trained using [LightGBM](https://github.com/microsoft/LightGBM). 
To enhance the first phase retrieval, 
a sequence-to-sequence neural network (T5) is used to perform document expansion with predicted queries. 
 
This initial baseline scores a MRR@100 of 0.355 on the **dev** and 0.312 on the **eval** set.
See [MS Marco Document Ranking Leaderboard](https://microsoft.github.io/MSMARCO-Document-Ranking-Submissions/leaderboard/).

# Vespa Document Schema
<pre>
schema doc {

  document doc {

    field title type string {
      indexing: summary | index
      index: enable-bm25
    }

    field text type string {
      indexing: summary | index
      index: enable-bm25
    }

    field url type string {
      indexing: summary | index
      index: enable-bm25
    }

    field id type string {
      indexing: summary |attribute
    }

    #Top-k prediction queries for this doc using T5 https://github.com/castorini/docTTTTTquery
    field doc_t5_query type array&lt;string&gt; {
      indexing: summary | index
      index: enable-bm25
    }
  }
} 
</pre> 

The original text fields plus an additional field from docTTTTTQuery is used by the retriever,
see Rodrigo Nogueira and Jimmy Lin paper: 
[From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf). 
The authors have published their pre-generated model and we use their predictions directly as is.
The only difference from the original paper and implementation
is that suggested expansion queries are indexed as an array instead of a blob of text. 
This allows calculating ranking features which takes proximity into account. 

# Training (LTR)
This work uses the MS Marco train split to scrape features for learning to rank. 

For each positive relevant document a sample of 50 negatives (not relevant) is picked from the top-k retrieved documents
using the baseline BM25 function. This ensures that the re-ranking model is trained on the distribution of the retriever. 

In total 330,302 queries from the training set is used , with 16,845,191 total number of data points. 
The efficient [Vespa WeakAnd](https://docs.vespa.ai/en/using-wand-with-vespa.html)
query operator is used to retrieve efficiently without having to score all documents matching at least one of the query terms. 

A set of 15 [ranking features](https://docs.vespa.ai/en/reference/rank-features.html)
which are generally cheap to compute are used by the model.

LightGBM is used to train the GBDT re-ranking model. 
Vespa has great support for GBDT models and supports both 
[LightGBM](https://docs.vespa.ai/en/lightgbm.html) and [XGBoost](https://docs.vespa.ai/en/xgboost.html). 

## LightGBM hyper-parameters

<pre>
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'eval_at': '5,10',
    'label_gain': [0,1],
    'lambdarank_truncation_level': 10,
    'eta':0.05,
    'num_leaves': 128,
    'min_data_in_leaf': 100,
    'feature_fraction':0.8
    }
</pre>

The final GBDT model consists of 533 trees, with up to 128 leaves. 
The training script is [here](src/main/python/train.py). 
T
o scrape features one can follow
[pyvespa collecting training data](https://pyvespa.readthedocs.io/en/latest/collect-training-data.html).

### Training output
<pre>
[LightGBM] [Info] Total Bins 2505
[LightGBM] [Info] Number of data points in the train set: 16845190, number of used features: 15
[1]     training's ndcg@5: 0.749873     training's ndcg@10: 0.770215    valid_1's ndcg@5: 0.353644      valid_1's ndcg@10: 0.391714
Training until validation scores don't improve for 50 rounds
[2]     training's ndcg@5: 0.783261     training's ndcg@10: 0.800121    valid_1's ndcg@5: 0.392956      valid_1's ndcg@10: 0.431421
[3]     training's ndcg@5: 0.792524     training's ndcg@10: 0.808461    valid_1's ndcg@5: 0.402466      valid_1's ndcg@10: 0.441569
[4]     training's ndcg@5: 0.795773     training's ndcg@10: 0.811582    valid_1's ndcg@5: 0.411093      valid_1's ndcg@10: 0.448714
[5]     training's ndcg@5: 0.797613     training's ndcg@10: 0.813361    valid_1's ndcg@5: 0.414653      valid_1's ndcg@10: 0.453241
[6]     training's ndcg@5: 0.799837     training's ndcg@10: 0.81526     valid_1's ndcg@5: 0.413973      valid_1's ndcg@10: 0.453484
[7]     training's ndcg@5: 0.801014     training's ndcg@10: 0.816459    valid_1's ndcg@5: 0.414625      valid_1's ndcg@10: 0.45447
[8]     training's ndcg@5: 0.801479     training's ndcg@10: 0.816772    valid_1's ndcg@5: 0.416506      valid_1's ndcg@10: 0.456094
[9]     training's ndcg@5: 0.802159     training's ndcg@10: 0.81744     valid_1's ndcg@5: 0.415425      valid_1's ndcg@10: 0.456132
</pre>

The serialized LightGBM model is deployed for serving using the following ranking profile. 
Re-ranking depth is set to 1K (per content node)
Note that the ranking profile inherits the ranking profile which used for feature scraping,
this avoids feature calculation drift, since the deployed model uses
the exact same feature definition as was used during training. 

<pre>
rank-profile ltr inherits ltr-scrape {
    num-threads-per-search: 12
    second-phase {
      expression: lightgbm("docranker.json")
      rerank-count: 1000
    }
  }
</pre>


# Ranking Evaluation 
See Vespa on the [MS Marco Document Ranking Leaderboard](https://microsoft.github.io/MSMARCO-Document-Ranking-Submissions/leaderboard/)

  **MS Marco Judgements** 

The official metric on MS Marco is [MRR@100](https://en.wikipedia.org/wiki/Mean_reciprocal_rank). 
 
**Dev MRR@100 = 0.355**,
**Eval MRR@100 = 0.312** 

A baseline bm25 model has MRR@100 around 0.161 on the Eval set.

# Run Time Serving Performance
Vespa's evaluation of GBDT models is hyper optimized after 20 years of using GBDT at scale 
so end to end serving time is roughly 20 ms. 
 
Vespa supports using multiple threads per *query* and in our experiment we use up to 12 threads per query. 


# Query API Example 
See the top two documents ranked for the question *when was nelson mandela born* below.

![Vespa Response for when was nelson mandela born](img/screen.png)


# Quick start

Make sure to read and agree to the terms and conditions of the 
[MS Marco Team](https://microsoft.github.io/msmarco/) before downloading the dataset by using the *ir_datasets* package. 

The following is a recipe on how to get started with a tiny set of sample data.
The sample data only contains the first 1000 documents of the full MS Marco dataset,
but this should be able to run on for instance a laptop.
For the full dataset to recreate the evaluation results, see later section.

Requirements:

* [Docker](https://www.docker.com/) Desktop installed and running. 6GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [Github releases](https://github.com/vespa-engine/vespa/releases).
* [Java 11](https://openjdk.java.net/projects/jdk/11/) installed.
* [Apache Maven](https://maven.apache.org/install.html) This sample app uses custom Java components and Maven is used
  to build the application.
* zstd: `brew install zstd`

Validate Docker resource settings, should be minimum 6GB:

<pre>
$ docker info | grep "Total Memory"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html).

<pre >
$ brew install vespa-cli
</pre>

Set target env, it's also possible to deploy to [Vespa Cloud](https://cloud.vespa.ai/)
using target cloud.

For local deployment using docker image use

<pre data-test="exec">
$ vespa config set target local
</pre>

For cloud deployment using [Vespa Cloud](https://cloud.vespa.ai/) use

<pre>
$ vespa config set target cloud
$ vespa config set application tenant-name.myapp.default
$ vespa auth login 
$ vespa auth cert
</pre>

See also [Cloud Vespa getting started guide](https://cloud.vespa.ai/en/getting-started). It's possible
to switch between local deployment and cloud deployment by changing the `config target`.

Pull and start the vespa docker container image:

<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Verify that configuration service (deploy api) is ready

<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application

<pre data-test="exec">
$ vespa clone msmarco-ranking myapp && cd myapp
</pre>

Download GBDT model which is used by [document ranking](document-ranking.md),
this step is required since both passage and document ranking is represented
in the same sample application.

<pre data-test="exec">
$ mkdir -p src/main/application/models
$ curl -L -o src/main/application/models/docranker.json.zst \
  https://data.vespa.oath.cloud/sample-apps-data/docranker.json.zst 
$ zstd -f -d src/main/application/models/docranker.json.zst 
</pre>

## Build the application package.
This step also downloads the three ONNX models used in this application package. The download
script used is found [here](src/main/bash/download_models.sh).

<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Make sure that the used Java version is 11.
The above mvn command will download models, build and package the vespa application package.

Deploy the application. This step deploys the application package built in the previous step:

<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

Wait for the application endpoint to become available

<pre data-test="exec">
$ vespa status --wait 300
</pre>

Running [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html)
which runs a set of basic tests to verify that the application is working as expected.
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/passage-ranking-system-test.json
</pre>
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/document-ranking-system-test.json
</pre>

## Feeding Sample Data
Feed the sample documents using the [vespa-feed-client](https://docs.vespa.ai/en/vespa-feed-client.html).

Download feeding client
<pre data-test="exec">
$ curl -L -o vespa-feed-client-cli.zip \
    https://search.maven.org/remotecontent?filepath=com/yahoo/vespa/vespa-feed-client-cli/7.527.20/vespa-feed-client-cli-7.527.20-zip.zip
$ unzip -o vespa-feed-client-cli.zip
</pre>

### Download sample feed files

<pre data-test="exec">
$ curl -L -o sample-feed/sample_regular_fields.jsonl.zst \
    https://data.vespa.oath.cloud/sample-apps-data/sample_regular_fields.jsonl.zst 

$ curl -L -o sample-feed/sample_doc_t5_query.jsonl.zst \
    https://data.vespa.oath.cloud/sample-apps-data/sample_doc_t5_query.jsonl.zst

$ zstd -d sample-feed/sample_regular_fields.jsonl.zst 

$ zstd -d sample-feed/sample_doc_t5_query.jsonl.zst 
</pre>

Feed the data

<pre data-test="exec">
$ ./vespa-feed-client-cli/vespa-feed-client \
    --file sample-feed/sample_regular_fields.jsonl --endpoint http://localhost:8080
</pre>

<pre data-test="exec">
$ ./vespa-feed-client-cli/vespa-feed-client \
    --file sample-feed/sample_doc_t5_query.jsonl --endpoint http://localhost:8080
</pre>

Now all the data is in place and one can play around with the query interface (Though only searching 1K documents).

View a sample document:
<pre data-test="exec" data-test-assert-contains="what is machacado">
$ curl -s http://localhost:8080/document/v1/msmarco/doc/docid/D1840066 | \
    python3 -m json.tool
</pre>

Do a query:
<pre data-test="exec" data-test-assert-contains="0.153">
$ curl -s "http://localhost:8080/search/?query=what%20is%20the%20definition%20of%20business%20law?&ranking=ltr&restrict=doc" | \
    python3 -m json.tool
</pre>

The data set is small, but one gets a feel for how the data and how the document to query expansion work. 
Note that negative relevance scores from the GBDT evaluation is normal. 

Shutdown and remove the Docker container:
<pre data-test="after">
$ docker rm -f vespa
</pre>


## Full Evaluation 
Using full dataset, all 3.2M documents.
Download and index the entire data set, including the document to query expansion. 

<pre>
$ python3 -m pip install ir_datasets tqdm requests
</pre>

### Download all documents 
<pre>
$ ir_datasets export msmarco-document/train docs --format jsonl | \
  ./src/main/python/document-feed.py > all-feed.jsonl
</pre>


## Document to query document expansion
For document expansion we use [docTTTTTquery](https://github.com/castorini/docTTTTTquery) 
Follow the instructions at 
[https://github.com/castorini/docTTTTTquery#per-document-expansion](https://github.com/castorini/docTTTTTquery#per-document-expansion),
but replace *paste -d" "* with *paste -d"#"*
and modify the *generate_output_dict* in *convert_msmarco_doc_to_anserini.py* to emit Vespa json instead:

<pre>
def generate_output_dict(doc, predicted_queries):
    doc_id = doc[0]
    preds = []
    for s in predicted_queries:
      s = s.strip().split("#")
      for k in s:
        preds.append(k)
    update = {
      "update": "id:msmarco:doc::{}".format(doc_id),
      "fields": {
        "doc_t5_query": {
         "assign": preds
        }
      }
    }
    return update
</pre>

Then run the script with --output_docs_path *doc_t5_query_updates.jsonl*.

Place the output file *doc_t5_query_updates.json* in the directory of the sample app. ($SAMPLE_APP)

<pre>
$ ./vespa-feed-client-cli/vespa-feed-client --file all-feed.jsonl --endpoint http://localhost:8080
</pre>

<pre>
$ ./vespa-feed-client-cli/vespa-feed-client --file doc_t5_query_updates.jsonl --endpoint http://localhost:8080
</pre>


## Query Evaluation
The following script will run all queries from the MS Marco document ranking **dev** split. 
Change the endpoint to point to your Vespa instance.
Since MS Marco is using MRR@100 we fetch at most 100 hits.

<pre>
$ ./src/main/python/evaluate_run.py --retriever sparse --rank_profile ltr --query_split dev \
  --wand_field default --wand_hits 500 --phase_count 1000 --run_file ltr.run.txt
</pre>

Evaluate the run file *ltr.run.txt* by using the 
[official ms marco eval script](https://raw.githubusercontent.com/microsoft/MSMARCO-Document-Ranking-Submissions/main/eval/ms_marco_doc_eval.py).

<pre>
$ curl -L -o msmarco-docdev-qrels.tsv.gz \
  https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz
$ curl -L -o ms_marco_doc_eval.py \
  https://raw.githubusercontent.com/microsoft/MSMARCO-Document-Ranking-Submissions/main/eval/ms_marco_doc_eval.py
$ gunzip msmarco-docdev-qrels.tsv.gz
$ python3 ms_marco_doc_eval.py --run ltr.run.txt --judgments msmarco-docdev-qrels.tsv  
</pre>

The above with the full corpus and with doc_t5_query should produce the following on the *dev* split:
<pre>
Quantity of Documents ranked for each query is as expected. Evaluating
#####################
MRR @100: 0.3546129653558302
QueriesRanked: 5193
#####################
</pre>

If you want to alter the application and submit to the leaderboard
you can generate a run for the **eval** query split by: 

<pre>
$ ./src/main/python/evaluate_document_run.py --retriever sparse --rank_profile ltr \
  --query_split eval --wand_field default --wand_hits 500 --phase_count 1000 --run_file eval.run.txt           
</pre>

The **eval** set relevancy judgements are hidden. To submit to the MS MARCO document ranking see 
[this repo](https://github.com/microsoft/MSMARCO-Document-Ranking-Submissions).
