<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

# MS Marco Document Ranking 

This is the baseline model for MS Marco *Document* Ranking. 

This document our first baseline model using traditional lexical matching (sparse) and ranking. 

We use sequence-to-sequence neural network (T5) to perform document expansion. 
 
This initial baseline scores a MRR@100 of 0.355 on the **dev** and 0.312 on the **eval** set.
See [MS Marco Document Ranking Leaderboard](https://microsoft.github.io/MSMARCO-Document-Ranking-Submissions/leaderboard/).

# Vespa Document Schema 

We index all 3.2M documents from the [MS Marco](https://microsoft.github.io/msmarco/) Document ranking collection  using the 
following [Vespa schema](src/main/application/schemas/doc.sd): 

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

We use the original text fields plus an additional field from docTTTTTQuery, see Rodrigo Nogueira and Jimmy Lin paper: 
[From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf). 
The authors have published their pre-generated model and we use their predictions directly as is. The only difference from the original is that
 we we index queries as an array instead of a blob of text. This allows calculating ranking features which takes proximity into account. 

# Training 
We use the MS Marco Train split to scrape features for traditional LTR. This enables us to use a range of features and the we use the training data to 
learn the optimal combination of these features.  

For each positive relevant document we sample 50 negatives from the top-k retrieved, using a simple 
linear combination of 
bm25 scores for body text, doc_t5_query, title and url. In total 330,302 queries from the training set is used and 16,845,191 total number of data points. 
We use the efficient
[Vespa WeakAnd](https://docs.vespa.ai/en/using-wand-with-vespa.html) implementation to retrieve efficiently. 

We handpick 15 [ranking features](https://docs.vespa.ai/en/reference/rank-features.html) which are generally cheap to compute except 
nativeProximity, but we limit it to the rather short title field. We don't do any type of feature normalization or selection except from what LightGBM does. 

We use LightGBM to train our model since Vespa has great support for GBDT models ([LightGBM](https://docs.vespa.ai/en/lightgbm.html), [XGBoost](https://docs.vespa.ai/en/xgboost.html)). 
We tune hyper parameters by observing the performance
on the development set. We end up with the following hyper parameters and we train for up to 1K iterations with early stopping after 50 iterations if the held out dev set performance does not improve:
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
We end up with 533 trees, with up to 128 leaves. The training script is [here](src/main/python/train.py). We plan to publish the raw training features but this gives an idea how the model was trained. 
To scrap features one can follow [pyvespa collecting training data](https://pyvespa.readthedocs.io/en/latest/collect-training-data.html).

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

We deploy our serialized LigtGBM model for serving and evaluation using the following profile. We keep the same linear simple first-phase function as described earlier. 
We re-rank up to 1K top hits from the simple untrained first-phase ranking expression.

<pre>
rank-profile ltr inherits ltr-scrape {
    num-threads-per-search: 12
    second-phase {
      expression: lightgbm("docranker.json")
      rerank-count: 1000
    }
  }
</pre>

docranker.json is deployed with the Vespa application package and translated to Vespa's optimized GBDT model evaluation. 
See [docranker.json (25MB)](src/main/application/models/docranker.json)


# Ranking Evaluation 
See our entry on the [MS Marco Document Ranking Leaderboard](https://microsoft.github.io/MSMARCO-Document-Ranking-Submissions/leaderboard/)

  **MS Marco Judgements** 

The offical metric on MS Marco is [MRR@100](https://en.wikipedia.org/wiki/Mean_reciprocal_rank). 
 
**Dev MRR@100 = 0.355**,
**Eval MRR@100 = 0.312** 

A baseline bm25 model has MRR@100 around 0.161 on the Eval set. 
  
  
# Run Time Serving Performance
Vespa's evaluation of GBDT models is hyper optimized after 20 years of using GBDT at scale 
so end to end serving time is roughly 20 ms. 
Vespa supports using multiple threads per *query* 
and in our experiment we use up to 12 threads per query. 
This allows scaling latency per node and make use of multi-core cpu architectures efficiently.

See the top two documents ranked for the question *when was nelson mandela born* below.
The per hit relevance score is assigned by the GBDT model. We search 3.2M documents on a single node and single partition and the 
weakAnd retrieves about 23K hits and the top 1K of those are re-ranked using the GBDT function using the features.

![Vespa Response for when was nelson mandela born](img/screen.png)

# Reproducing this work 
We use the [IR_datasets](https://github.com/allenai/ir_datasets) python package to obtain the MS Marco Document and Passage ranking dataset.

Make sure to go read and agree to terms and conditions of [MS Marco Team](https://microsoft.github.io/msmarco/) before downloading the dataset by using the *ir_datasets* package. 

We also use the [LightGBM](https://lightgbm.readthedocs.io/en/latest/) library which is a gradient boosting framework that uses tree based learning algorithms. 

## Quick start

The following is a recipe on how to get started with a tiny set of sample data.
The sample data only contains the first 1000 documents of the full MS Marco dataset, but
this should be able to run on for instance a laptop. For the full dataset to
recreate the evaluation results see later section.

Requirements:

* [Docker](https://www.docker.com/) installed and running. 10Gb available memory for Docker is recommended.
* Git client to checkout the sample application repository
* Java 11, Maven and python3
* zstd: `brew install zstd`
* Operating system: macOS or Linux, Architecture: x86_64

See also [Vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html).

Clone the sample app 

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/msmarco-ranking
</pre>

<pre data-test="exec">
$ python3 -m pip install torch transformers ir_datasets lightgbm numpy pandas requests tqdm
$ mvn clean package -U
</pre>

<pre data-test="exec">
$ pip3 install torch numpy ir_datasets requests tqdm transformers onnx onnxruntime
</pre>

Since we use a shared application package for both [passage](passage-ranking.md) and document ranking 
we also need download models which are used by the passage ranking part of this sample app.
 
Download ONNX models which have been exported by us and made available for this sample app. 

<pre data-test="exec">
$ mkdir -p src/main/application/files/
</pre>

<pre data-test="exec">
$ curl -L -o src/main/application/files/sentence-msmarco-MiniLM-L-6-v3-quantized.onnx \
    https://data.vespa.oath.cloud/onnx_models/sentence-msmarco-MiniLM-L-6-v3-quantized.onnx
</pre>

<pre data-test="exec">
$ curl -L -o src/main/application/files/ms-marco-MiniLM-L-6-v2-quantized.onnx \
    https://data.vespa.oath.cloud/onnx_models/ms-marco-MiniLM-L-6-v2-quantized.onnx
</pre>

<pre data-test="exec">
$ curl -L -o src/main/application/files/vespa-colMiniLM-L-6-quantized.onnx \
    https://data.vespa.oath.cloud/onnx_models/vespa-colMiniLM-L-6-quantized.onnx
</pre>

Once we have downloaded the models, 
we use maven to create the
[Vespa application package](https://docs.vespa.ai/en/reference/application-packages-reference.html).

<pre data-test="exec">
$ mvn clean package -U
</pre>

If you run into issues running mvn package please check  mvn -v and that the Java version is 11. 
Now, we are ready to start the vespeengine/vespa docker container - pull the latest version and run it by

Start the Vespa docker container:

<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Wait for configuration service to start (the command below should return a 200 OK):

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:19071/ApplicationStatus
</pre>

Deploy the application package:

<pre data-test="exec" data-test-assert-contains="prepared and activated.">
$ curl --header Content-Type:application/zip --data-binary @target/application.zip \
  localhost:19071/application/v2/tenant/default/prepareandactivate
</pre>

Now, wait for the application to start.

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

## Feeding Sample Data 

Feed the sample documents using the [Vespa http feeder client](https://docs.vespa.ai/en/vespa-http-client.html):
<pre data-test="exec">
$ curl -L -o vespa-http-client-jar-with-dependencies.jar \
    https://search.maven.org/classic/remotecontent?filepath=com/yahoo/vespa/vespa-http-client/7.391.28/vespa-http-client-7.391.28-jar-with-dependencies.jar
</pre>

<pre data-test="exec">
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --file sample-feed/sample_regular_fields.jsonl --endpoint http://localhost:8080
</pre>

<pre data-test="exec">
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --file sample-feed/sample_doc_t5_query.jsonl --endpoint http://localhost:8080
</pre>

Now all the data is in place and one can play around with the query interface (Though only searching 1K documents)

View a sample document 
<pre>
$ curl -s http://localhost:8080/document/v1/msmarco/doc/docid/D1840066 | \
    python -m json.tool
</pre>

Do a query 
<pre>
$ curl -s "http://localhost:8080/search/?query=what%20is%20the%20definition%20of%20business%20law?&ranking=ltr" | \
    python -m json.tool
</pre>

The data set is small, but one gets a feel for how the data and how the document to query expansion work. 
Note that negative relevance scores from the GBDT evaluation is normal. 

## Full Evaluation (Using full dataset, all 3.2M documents)
First we need to download and index the entire data set and the document to query expansion. 

### Download all documents 
<pre>
$ ir_datasets export msmarco-document/train docs --format jsonl | \
  ./src/main/python/document-feed.py > all-feed.jsonl
</pre>

## doc to query document expansion
For document expansion we use [docTTTTTquery](https://github.com/castorini/docTTTTTquery) 
Follow the instructions at [https://github.com/castorini/docTTTTTquery#per-document-expansion](https://github.com/castorini/docTTTTTquery#per-document-expansion),
but replace *paste -d" "* with *paste -d"#"* and modify the *generate_output_dict* in *convert_msmarco_doc_to_anserini.py* to emit Vespa json instead 

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
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --file all-feed.jsonl --endpoint http://localhost:8080
</pre>

<pre>
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --file doc_t5_query_updates.jsonl --endpoint http://localhost:8080
</pre>

## Query Evaluation
The following script will run all queries from the MS Marco document ranking **dev** split. Change the endpoint to point to your Vespa instance. Since MS Marco is using MRR@100 we
fetch at most 100 hits.

<pre>
$ ./src/main/python/evaluate_run.py --retriever sparse --rank_profile ltr --query_split dev \
  --wand_field default --wand_hits 500 --phase_count 1000 --run_file ltr.run.txt
</pre>

We can evaluate the run file *ltr.run.txt* by using the [official ms marco eval script](https://raw.githubusercontent.com/microsoft/MSMARCO-Document-Ranking-Submissions/main/eval/ms_marco_doc_eval.py).

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

If you want to alter the application and submit to the leaderboard you can generate a run for the **eval** query split by 


<pre>
$ ./src/main/python/evaluate_document_run.py --retriever sparse --rank_profile ltr \
  --query_split eval --wand_field default --wand_hits 500 --phase_count 1000 --run_file eval.run.txt           
</pre>

The **eval** set relevancy judgements are hidden. To submit to the MS MARCO document ranking see 
[this repo](https://github.com/microsoft/MSMARCO-Document-Ranking-Submissions)
