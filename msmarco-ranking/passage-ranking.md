<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

# MS Marco Passage Ranking using ColBERT 

This sample appplication demonstrates how to efficiently represent 
[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) 
using Vespa.ai. There are three key features of Vespa which allows representing the ColBERT model efficiently on Vespa:

* The ability to run online inference using pretrained language models like BERT in Vespa using Vespa's [ONNX](https://docs.vespa.ai/en/onnx.html) support.
ColBERT only requires encoding the short query in user time.  
* Ability to store tensor representations in documents and in queries. Using a [tensor expressions](https://docs.vespa.ai/documentation/tensor-user-guide.html)
 we can express the ColBERT MaxSim operator of ColBERT in a Vespa ranking expression.
* [Multi-phase retrieval and ranking](https://docs.vespa.ai/en/phased-ranking.html) in Vespa allows expressing an efficient 
retriever which retrieves candidate documents which are re-ranked using ColBERT.

We deploy ColBERT on Vespa as a re-ranker, re-ranking the top K documents from a more efficient retriever and in our case we
use the sparse [weak And](https://docs.vespa.ai/documentation/using-wand-with-vespa.html) query operator to 
efficiently retrieve a candidate set of documents which are re-ranking using the ColBERT model. Vespa also allows dense retrieval accelerated 
by approximate nearest neighbor search. For example see [Dense Passage retrieval sample application](../dense-passage-retrieval-with-ann).


## About ColBERT

![Colbert overview](img/colbert_illustration.png)

*Illustration from [ColBERT paper](https://arxiv.org/abs/2004.12832)*

Unlike the *all to all* query document interaction model, the late interaction *ColBERT* model enables processing 
the documents offline since the per term contextual embedding is independent of the query. 

This significantly speeds up onstage query evaluation since at query time one just need to calculate the contextual query term embeddings and 
calculate the MaxSim operator over the pre-computed contextualized embeddings for the documents we want to re-rank. 

As demonstrated in the paper, the late interaction ColBERT model achieves almost the same ranking accuracy
as the more computationally complex *all to all* query document interaction models.   
 
For an in-depth overview of applying pre-trained Transformer language models for retrieval and ranking we  
can recommend [Pretrained Transformers for Text Ranking: BERT and Beyond](https://arxiv.org/abs/2010.06467). 

![Colbert MaxSim](img/colbert_illustration_zoom.png)

*The ColBERT Max Sim operator. Illustration from [ColBERT paper](https://arxiv.org/abs/2004.12832)*

The ColBERT model architecture consists of two encoders, a query encoder and a document encoder which both are built over the same BERT model weights.
The max sequence length of the input document is configurable (up to the 512 max limit of BERT).  During offline processing and indexing one 
can compute the per term contextual embeddings and store the resulting tensor for re-ranking.  

During offline processing of the corpus, each term in the passage text is represented by a vector of n dimensions.
The embedding  representation per term is dependent of the other terms in the passage (attention mechanism),hence contextualized. 

The dimensionality used impacts ranking accuracy but also memory footprint, the authors demonstrates that the accuracy is not impacted significantly 
by reducing the number of dimensions from 128 to 64 and in our experiments with ColBERT we reduce the dimensionality to 32 per term  


## MS Marco Passage Ranking Evaluation 

The official ranking evaluation metric on MS Marco passage leaderboard is [MRR@10](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 
and below are the results of our ColBERT implementation on the **eval** and **dev** query set. 
The MS Marco Passage ranking leaderboard is sorted by the score on the *eval* set. 
See [MSMarco Passage Ranking Leaderboard](https://microsoft.github.io/msmarco/)
 
For reference we also include 
the official baseline using [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and 
a tuned BM25 using *Apache Lucene 8* which powers text ranking search for engines like *Apache Solr* and *Elasticsearch*. 

| Model                    | Eval  | Dev   |
|--------------------------|-------|-------|
| BM25 (Official baseline) | 0.165 | 0.167 |
| BM25 (Lucene8, tuned)    | 0.190 | 0.187 |
| **ColBERT on Vespa.ai**  | 0.347 | 0.354 |

Note that the effectiveness depends on the retriever. In our sample run we used BM25 which has about 0.85 recall@1K. The retriever effectiveness sets
the upper bound of the re-rankers effectiveness and training ColBERT on a dense retriever with a higher recall (e.g 0.97@1K) would likely produce a better re-ranker. 

The model was trained on the original positive and negative examples, we believe that the ranking metrics could be improved by using a dense retriever 
with better recall and instead sampling hard negatives from the distribution of the dense retriever. 


## ColBERT representation on Vespa
 
Vectors or generally tensors are first-class citizens in the Vespa document model and we use the following 
Vespa [passage](src/main/application/schemas/passage.sd) document schema with 3 fields:

<pre>
schema passage {

  document passage {

    field text type string {
      indexing: summary | index
      index: enable-bm25
    }
    
    field dt type tensor&lt;float&gt;(dt{}, x[32]){
      indexing: summary | attribute
    }

    field id type int {
      indexing: summary |attribute
    }
  }
} 
</pre> 

The *text* field contains the original passage text and the *dt* tensor field stores the contextual term embedding. *dt* is an example of a mixed tensor
which mixes sparse ("dt") and dense "x". Using sparse allows us to store as many per term embeddings as find in the document. 

We use 32 dimensions for the contextualized term embedding and this is denoted by <em>x[32]</em>. The max length is capped at 180 tokens for the document passage. 
 
See [Vespa Tensor Guide](https://docs.vespa.ai/en/tensor-user-guide.html) for an introduction to Vespa tensors. The *id* field is the passage id from the 
dataset. This id is used to evaluate the ranking accuracy. We also define a *id* document summary which avoids returning the text and the potential large *dt* tensor field. 

We define two ranking profile in our *passage* document schema. See [Vespa Ranking Documentation](https://docs.vespa.ai/en/ranking.html)
for an overview of how to represent ranking in Vespa.

The baseline ranking model is using [bm25](https://docs.vespa.ai/documentation/reference/bm25.html). BM25 could be 
a good baseline model for many domains where there are no prior training data available. Be it by explicit relevancy judgements 
like in MS Marco or implicit through user feedback (click or dwell time) used to train a dense retriever. Below is the the *bm25* ranking profile:

<pre> 
rank-profile bm25 {
  num-threads-per-search: 6
  first-phase {
    expression: bm25(text)
  }
}
</pre>
The bm25 ranking feature has two hyperparameters and we leave them at the default in Vespa (b=0.75, k=1.2). We specify 
that retrieval and ranking should be using 6 threads per search. Vespa supports using multiple threads to evaluate a query which allows tuning latency
versus throughput. Not all retrieval and ranking cases needs a lot of throughput but end to end latency needs to meet service latency agreement
and controlling the number of threads per search query can help. 
See [Vespa performance and sizing guide](https://docs.vespa.ai/documentation/performance/sizing-search.html) 

 
We define the *colBERT* reranker with a Vespa ranking profile which uses [phased ranking](https://docs.vespa.ai/en/phased-ranking.html). 
The first phase is inherited from the *bm25* ranking profile. The way this work is that the top K documents as scored by 
 the bm25(text) ranking feature is re-ranked using the ColBERT MaxSim operator. The re-ranking count or re-ranking depth is configurable 
 and can also be overridden at query time.
 
The ColBERT MaxSim operator is expressed using Vespa's
[tensor expression language](https://docs.vespa.ai/en/reference/ranking-expressions.html#tensor-functions). 

 
<pre> 
rank-profile colbert inherits bm25 {
  num-threads-per-search: 6
  second-phase {
    rerank-count: 1000
    expression {
      sum(
        reduce(
          sum(
            query(qt) * attribute(dt), x
          ),
          max, dt
        ),
        qt
      )
    }
  }
}
</pre>


The *query(qt)* represent the ColBERT query tensor which is computed online first before retrieving and ranking passages.  
 The query tensor hold the per term contextual embeddings. 
 *attribute(dt)* reads the document tensor which stores the per term contextual embedding. Tensor attribute fields in Vespa are in-memory.  


The query tensor type is defined in the application package in
[src/main/application/search/query-profiles/types/root.xml](src/main/application/search/query-profiles/types/root.xml):
<pre>
&lt;field name="ranking.features.query(qt)" type="tensor&lt;float&gt;(qt{},x[32])"/&gt;
</pre>

The *query(qt)* tensor is as we can see also a mixed tensor.  

We represent the ColBERT query encoder in Vespa 
using Vespa's support for inference and [ranking with ONNX](https://docs.vespa.ai/en/onnx.html) models. 
In this case we don't rank any documents directly with the ONNX model but we use the support to make a single pass through the ColBERT query encoder. 
We plug the model into the Vespa serving architecture using a *query* document type which is empty as it's only used to represent the query encoder *model*.

We use a dedicated *query* document type for the query encoder (Notice that the query document schema is empty).

<pre> 
schema query {
    document query {
    }
    onnx-model encoder {
      file: files/colbert_query_encoder.onnx
      input  input_ids: query(input_ids)
      input  attention_mask: query(attention_mask)
      output contextual:contextual 
    }

    rank-profile colbert_query_encoder {
      num-threads-per-search: 1
      first-phase {
        expression: random 
      }
      summary-features {
        onnxModel(encoder).contextual
      }
    }
}
</pre>
We feed one empty *query* document and use the standard search and summary fill to compute the contextual embeddings of the Vespa query by invoking 
the *colbert_query_encoder* ranking profile over the single retrieved document. 


## ColBERT - Model training and offline text to tensor processing 

* We train the *ColBERT* model using the instructions from the [ColBERT repository](https://github.com/stanford-futuredata/ColBERT) using the MS Marco Passage training set.
 We replace the *bert-base-uncased* with the *bert-medium-uncased*
as our base model. The medium BERT< model is faster to run inference over and we can also train with a larger batch size. 
The [bert-medium-uncased](https://huggingface.co/google/bert_uncased_L-8_H-512_A-8) has 
8 layers and a hidden dimensionality of 512. We use cosine similarity (innerproduct as the vectors are unit length normalized).
* The dimensionality of the token tensor is reduced from 512 (hidden dim) to 32 dimensions by a linear layer. 
We limit the query length to 32 subword tokens and passage text to 80 tokens. This is the default in ColBERT.
* We use the GPU powered indexing routine in the mentioned *ColBERT repository* to obtain the document tensor representation
* We export the ColBERT query encoder to [ONNX](https://onnx.ai/) format for serving in Vespa.ai using the ONNX runtime which Vespa integrates with. 
See [Ranking with ONNX Models](https://docs.vespa.ai/documentation/onnx.html) and 
[How we accelerate inference using ONNX runtime](https://blog.vespa.ai/stateful-model-serving-how-we-accelerate-inference-using-onnx-runtime/)
* The trained weights of our model is published on [Hugginface model hub](https://huggingface.co/vespa-engine/colbert-medium) and 
we provide instructions on howto export the Transformer model to ONNX format 


## Scaling and Serving Performance
See [Scaling and performance evaluation of ColBERT on Vespa.ai](colbert-performance-scaling.md)


## Further work 

* Use bfloat16 or int8 for the ColBERT document tensors to save memory. The current full model takes about 138GB of memory when loaded in a Vespa instance.
* Replace the sparse wand based retrieved with a dense retriever with higher recall@1K and train ColBERT on the distribution of the dense retriever instead 


# Reproducing this work 
Make sure to go read and agree to terms and conditions of [MS Marco Team](https://microsoft.github.io/msmarco/) 
before downloading the dataset. Note that the terms and conditions does not permit using the MS Marco Document/Passage data for commercial use.

## Quick start

The following is a quick start recipe on how to get started with a tiny set of sample data.
The sample data only contains the first 1000 documents of the full MS Marco passage ranking dataset. We use pre-computed document ColBERT tensors.

This should be able to run on for instance a laptop. For the full dataset to reproduce our submission to the MS Marco Passage ranking leaderboard see 
[full evaluation](#full-evaluation).

Requirements:

* [Docker](https://www.docker.com/) installed and running. 10Gb available memory for Docker is recommended.
* Git client to checkout the sample application repository
* Java 11, Maven and python3 installed, curl and wget in $PATH
* Operating system: macOS or Linux, Architecture: x86_64

See also [Vespa quick start guide](https://docs.vespa.ai/documentation/vespa-quick-start.html).

First, we clone the sample apps :

<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ SAMPLE_APP=`pwd`/sample-apps/msmarco-ranking
$ cd $SAMPLE_APP
</pre>

Install python dependencies. There are no run time python dependencies in Vespa but we use. 

<pre data-test="exec">
$ pip3 install torch numpy ir_datasets requests tqdm transformers onnx onnxruntime
</pre>

The *model_export.py* script downloads the pre-trained weights from [Huggingface model hub](https://huggingface.co/vespa-engine/colbert-medium)
and exports the ColBERT query encoder to ONNX format for serving in Vespa:
 
<pre data-test="exec">
$ mkdir src/main/application/files/
$ python3 src/main/python/model_export.py src/main/application/files/colbert_query_encoder.onnx 
</pre>

Once we have downloaded and exported the model we use maven to create the

 [Vespa application package](https://docs.vespa.ai/en/reference/application-packages-reference.html).


<pre data-test="exec">
$ mvn clean package -U
</pre>

If you run into issues running mvn package please check  mvn -v and that the Java version is 11. 


Now, we are ready to start the vespeengine/vespa docker container. We pull the latest version and run it by 

<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $SAMPLE_APP:/MSMARCO --publish 8080:8080 vespaengine/vespa
</pre>

Wait for configuration service to start, the command below should return a 200 OK:

<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

Deploy the application package:

<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /MSMARCO/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>

now, wait for the application to start, the command below should return a 200 OK

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

## Feeding Sample Data 

We feed the sample documents using the [Vespa http feeder
client](https://docs.vespa.ai/en/vespa-http-client.html).

Download the sample data

<pre data-test="exec">
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-sample.jsonl.zst\
    -O sample-feed/colbert-passages-sample.jsonl.zst
</pre>

Feed the data 

<pre data-test="exec">
$ docker exec vespa bash -c 'zstdcat /MSMARCO/sample-feed/colbert-passages-sample.jsonl.zst| java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
     --host localhost --port 8080'
</pre>

Feed the query document:

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /MSMARCO/sample-feed/sample_query_feed.jsonl --host localhost --port 8080'
</pre>

Now all the data is indexed and one can play around with the search interface. Note, only searching 1K passages.

For example do a query for *what was the Manhattan Project*:

<pre>
$ cat sample-feed/query.json
{
    "yql": "select id, text from sources passage where userQuery();",
    "ranking": "colbert",
    "query": "what was the manhattan project?",
    "hits": 5
}
</pre>

<pre data-test="exec" data-test-wait-for="200 OK">
curl -s -H "Content-Type: application/json" --data @sample-feed/query.json \
 http://localhost:8080/search/ |python3 -m json.tool
</pre>

It is also possible to view the result in a 
[browser](http://localhost:8080/search/?yql=select%20id,%20text%20from%20sources%20passage%20where%20userQuery()%3B&query=what+was+the+Manhattan%20Project&ranking=colbert&searchChain=passageranking)

![Colbert example](img/colbert_sample_query.png)

One can also compare ranking with the *bm25* ranking profile: 

<pre>
{
    "yql": "select id, text from sources passage where userQuery();",
    "ranking": "bm25",
    "query": "what was the manhattan project?",
    "hits": 5,
    "searchChain": "passageranking" 
}
</pre>



## Full Evaluation

### Download all passages 

Feed query document (which allows the evaluation of the ColBERT query encoder):

<pre>
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /MSMARCO/sample-feed/sample_query_feed.jsonl	--host localhost --port 8080'
</pre>

### Download pre-processed ColBERT document representation 
Download the preprocessed document tensor data. The data is compressed using [ZSTD](https://facebook.github.io/zstd/): 

<pre>
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-1.jsonl.zst  -O sample-feed/colbert-passages-1.jsonl.zst
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-2.jsonl.zst  -O sample-feed/colbert-passages-2.jsonl.zst
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-3.jsonl.zst  -O sample-feed/colbert-passages-3.jsonl.zst
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-4.jsonl.zst  -O sample-feed/colbert-passages-4.jsonl.zst
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-5.jsonl.zst  -O sample-feed/colbert-passages-5.jsonl.zst
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-6.jsonl.zst  -O sample-feed/colbert-passages-6.jsonl.zst
</pre>

Note that we stream through the data using *zstdcat* as the uncompressed representation
is large (300GB). 

<pre>
$ docker exec vespa bash -c 'zstdcat /MSMARCO/sample-feed/colbert-passages-*.zst | java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
     --host localhost --port 8080'
</pre>

### Ranking Evaluation using Ms Marco Passage Ranking 
Run through all queries from the MS Marco Passage Ranking Dev set

<pre>
./src/main/python/evaluate_passage_run.py --query_split dev --retriever sparse --rank_profile colbert --hits 10 --run_file run.dev.txt
</pre>

To evaluate ranking performance download the official evaluation scripts

<pre>
wget https://raw.githubusercontent.com/spacemanidol/MSMARCO/master/Ranking/Baselines/msmarco_eval.py
</pre>

Generate the dev qrels file using the *ir_datasets* which the evaluation script expects:
<pre>
./src/main/python/dump_passage_dev_qrels.py 
</pre>

Above will write a **qrels.dev.small.tsv** file to the current directory, now we can run evaluation:
<pre>
python3 msmarco_eval.py qrels.dev.small.tsv run.dev.txt 
#####################
MRR @10: 0.3540263564833764
QueriesRanked: 6980
#####################
</pre>

To generate runs using the eval set pass *--query_split* eval: 
<pre>
./src/main/python/evaluate_passage_run.py --query_split eval --retriever sparse --rank_profile colbert --hits 10 --run_file run.eval.txt
</pre>
The *eval* split is the hold out test set where there are no available judgments in the public domain. 
See [MS Marco Passage Ranking](https://microsoft.github.io/MSMARCO-Passage-Ranking/) for how to submit runs for the leaderboard. 
