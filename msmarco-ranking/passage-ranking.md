<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

# MS Marco Passage Ranking using ColBERT 

This sample appplication demonstrates how to efficiently represent 
[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) 
using Vespa.ai. There are three key features of Vespa which allows representing the ColBERT model efficiently on Vespa:

* The ability to run inference using pretrained language models like BERT in Vespa using Vespa's [ONNX](https://docs.vespa.ai/en/onnx.html) support.
ColBERT only requires encoding the short query in user time.  
* Ability to store tensor representations in documents and in queries. Using a [tensor expressions](https://docs.vespa.ai/documentation/tensor-user-guide.html) we can express the ColBERT MaxSim operator of ColBERT in a Vespa ranking expression.
* [Multi-phase retrieval and ranking](https://docs.vespa.ai/en/phased-ranking.html) in Vespa allows expressing a efficient retriever which retrieves candidate documents which are re-ranked using ColBERT.

We deploy ColBERT on Vespa as a re-ranker, re-ranking the top K documents from a more efficient retriever and in our case we
use the [weak And](https://docs.vespa.ai/documentation/using-wand-with-vespa.html) query operator to 
efficiently retrieve a candidate set of documents which are re-ranking using the ColBERT model. 


## About ColBERT

![Colbert overview](img/colbert_illustration.png)

*Illustration from [ColBERT paper](https://arxiv.org/abs/2004.12832)*

Unlike the *all to all* query document interaction model, the late interaction *ColBERT* model enables processing the documents offline since the per term contextual 
embedding is independent of the query. This significantly speeds up query evaluation since at query time we only need to calculate the contextual query term embeddings and finally
calculate the MaxSim operator over the pre-computed contextualized embeddings for the documents we want to re-rank. 

As demonstrated in the paper, the late interaction ColBERT model achieves almost the same ranking accuracy
as the more computationally complex *all to all* query document interaction models.   
 
For an in-depth overview of applying pretrained Transformer language models for text ranking we  
can recommend [Pretrained Transformers for Text Ranking: BERT and Beyond](https://arxiv.org/abs/2010.06467). 

![Colbert MaxSim](img/colbert_illustration_zoom.png)

*The MaxSim operator, Illustration from [ColBERT paper](https://arxiv.org/abs/2004.12832)*

The ColBERT model architecture consists of two encoders, a query encoder and a document encoder which both are built over the same BERT model weights.
The max sequence length of the input document is configurable (up to the 512 max limit of BERT).  During offline processing and indexing one 
can compute the per term contextual embeddings and store the resulting tensor for re-ranking.  

During offline processing of the corpus, each term in the passage is represented by a vector of n dimensions.
The vector representation per term is dependent of the other terms in the passage (attention mechanism),hence contextualized. 
The embedding of each term depends on the company it keeps in the passage text. 
The dimensionality used impacts ranking accuracy but also memory footprint, the authors demonstrates that the accuracy is not impacted significantly 
by reducing the number of dimensions from 128 to 64. 


## Ranking Evaluation 

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

Note that the effectiveness depends on the retriever. In our sample run we used wand sparse retrieval which has about 0.8 recall@1K. The retriever sets
the upper bound of the re-rankers effectiveness and training ColBERT on a dense retriever with a higher recall (e.g 0.97@1K) would produce a better re-ranker. 


## Representing ColBERT on Vespa.ai
 
Vectors or generally tensors are first-class citizens in the Vespa document model and we use the following Vespa *passage* document schema with
3 fields:

<pre>
schema passage {

  document passage {

    field text type string {
      indexing: summary | index
      index: enable-bm25
    }
    
    field dt type tensor<float>(dt{}, x[32]){
      indexing: summary | attribute
    }

    field id type int {
      indexing: summary |attribute
    }
  }
} 
</pre> 

The *text* field contains the original passage text and the *dt* tensor field stores the contextual term embedding. We use 32 dimensions for the contextual embedding. 
See [Vespa Tensor Guide](https://docs.vespa.ai/en/tensor-user-guide.html) for an introduction to Vespa tensors. The *id* field is the passage id from the 
dataset. This id is used to evaluate the accuracy.  We use Vespa's support for [phased ranking](https://docs.vespa.ai/en/phased-ranking.html)


We define two ranking profile in our *passage* schema. See [Vespa Ranking](https://docs.vespa.ai/en/ranking.html). 
We first retrieve candidate documents using term based matching ranking with bm25 and we re-rank the top 1K documents from 
first-phase using the ColBERT MaxSim operator which here is expressed using Vespa's
[tensor expression language](https://docs.vespa.ai/en/reference/ranking-expressions.html#tensor-functions). 

  
<pre> 
rank-profile bm25 {
  first-phase {
    expression: bm25(text)
  }
}
  
rank-profile colbert inherits bm25 {
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

The *query(qt)* represent the colbert query tensor with the per query term contextual embeddings. *attribute(dt)* 
reads the document tensor which stores the per term contextual embedding. Attribute fields in Vespa are in-memory fields. 


The query tensor type is defined in the application package in
[src/main/application/search/query-profiles/types/root.xml](src/main/application/search/query-profiles/types/root.xml):
<pre>
&lt;field name="ranking.features.query(qt)" type="tensor&lt;float&gt;(qt{},x[32])"/&gt;
</pre>

The *query(qt)* tensor is the resulting tensor of the ColBERT query encoder which needs to be computed online. 
We represent the ColBERT query encoder in Vespa 
using Vespa's support for inference and [ranking with ONNX](https://docs.vespa.ai/en/onnx.html) models. 
In this case we don't rank any documents directly with the ONNX model.  
We plug the model into the Vespa serving architecture using a *query* document type which is empty as it's only used to represent the query encoder *model*.

The inputs to the model are also query tensors and defined in the 
[tensor types](src/main/application/search/query-profiles/types/root.xml). 

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
We feed one empty *query* document and use the standard search and summary fill to compute the contextual embeddings of the Vespa query. 


## ColBERT - Model training and offline processing 

* We train the *ColBERT* model using the instructions from the [ColBERT repository](https://github.com/stanford-futuredata/ColBERT) using the MS Marco Passage training set.
 We replace the *bert-base-uncased* with the *bert-medium-uncased*
as our base model. The medium BERT< model is faster to run inference over and we can also train with a larger batch size. 
The [bert-medium-uncased](https://huggingface.co/google/bert_uncased_L-8_H-512_A-8) has 
8 layers and a hidden dimensionality of 512. We use cosine similarity (innerproduct) and the dimensionality of the token tensor is reduced from 512 to 32 dimensions. 
We limit the query length to 32 subword tokens and passage text to 80 tokens. This is the default in ColBERT.
* We use the GPU powered indexing routine in the mentioned *ColBERT repository* to obtain the document representation with a dense vector per token_id in from the passage text.
* We export the ColBERT query encoder to [ONNX](https://onnx.ai/) format for serving in Vespa.ai using the ONNX runtime which Vespa integrates with. 
See [Ranking with ONNX Models](https://docs.vespa.ai/documentation/onnx.html) and 
[How we accelerate inference using ONNX runtime](https://blog.vespa.ai/stateful-model-serving-how-we-accelerate-inference-using-onnx-runtime/)
* The trained weights of our model is published on [Hugginface model hub](https://huggingface.co/vespa-engine/colbert-medium) and 
we provide instructions on howto export the Transformer model to ONNX format 
* We use the [Vespa tensor expression](https://docs.vespa.ai/documentation/reference/ranking-expressions.html#tensor-functions) 
support to express the ColBERT MaxSim operator which computes 32*N inner dot products per passage 
where N is the number of tokens in the passage (up to max 80).  

## Serving Performance


# Reproducing this work 

We use the [IR_datasets](https://github.com/allenai/ir_datasets) python package to obtain the MS Marco Document and Passage ranking dataset.

Make sure to go read and agree to terms and conditions of [MS Marco Team](https://microsoft.github.io/msmarco/) 
before downloading the dataset by using the *ir_datasets* package. 

## Quick start

The following is a recipe on how to get started with a tiny set of sample data.
The sample data only contains the first 1000 documents of the full MS Marco passage ranking dataset. 

This should be able to run on for instance a laptop. For the full dataset to reproduce our submission to the leaderboard see 
[full evaluation](#full-evaluation).

Requirements:

* [Docker](https://www.docker.com/) installed and running. 10Gb available memory for Docker is recommended.
* Git client to checkout the sample application repository
* Java 11, Maven and python3 installed, curl and wget in $PATH
* Operating system: macOS or Linux, Architecture: x86_64

See also [Vespa quick start guide](https://docs.vespa.ai/documentation/vespa-quick-start.html).

First, we retrieve the sample app:

<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ SAMPLE_APP=`pwd`/sample-apps/msmarco-ranking
$ cd $SAMPLE_APP
</pre>

Install python dependencies. There are no run time python dependencies in Vespa. 

<pre data-test="exec">
$ pip3 install torch ir_datasets requests tqdm transformers
</pre>

The model_export download the pre-trained weights from [Huggingface](https://huggingface.co/vespa-engine/colbert-medium) and export 
the ColBERT query encoder to ONNX format for serving in Vespa:
 
<pre data-test="exec">
$ python3 src/main/python/model_export.py src/main/application/files/colbert_query_encoder.onnx 
</pre>
The *mode_export.py* script downloads the model from Hugginface and exports it to ONNX for serving.

The maven clean package will build the Vespa application package file (*target/application.zip*) 
which is later used when we have started the Vespa services.

<pre data-test="exec">
$ mvn clean package -U
</pre>

Start the Vespa docker container:

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

Feed sample data. 

We feed the passage documents using the [Vespa http feeder
client](https://docs.vespa.ai/en/vespa-http-client.html):

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /MSMARCO/sample-feed/sample_passage_feed.jsonl --host localhost --port 8080'
</pre>

Feed the empty query document type 
<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /MSMARCO/sample-feed/sample_query_feed.jsonl	--host localhost --port 8080'
</pre>

Download pre-computed sample for the first 1K documents
<pre data-test="exec">
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-tensor-update-sample.jsonl \
    -O sample-feed/colbert-tensor-update-sample.jsonl
</pre>

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /MSMARCO/sample-feed/colbert-tensor-update-sample.jsonl --host localhost --port 8080'
</pre>


Now all the data is in place and one can play around with the search interface (Though only searching 1K documents)

View a sample document 
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s http://localhost:8080/document/v1/msmarco/passage/docid/0 |python3 -m json.tool
</pre>

Do a query for *what was the Manhattan Project*:

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
    "hits": 5
}
</pre>



## Full Evaluation

To reproduce with the full dataset one needs disk space to store the tensor data and enough memory. 

Details to follow.

### Download all passages 

Download and process the entire passage data set using the **ir_datasets** export tool. 

<pre>
$ ir_datasets export msmarco-passage/train docs --format jsonl  |./src/main/python/passage-feed.py > sample-feed/passage-all-feed.jsonl
</pre>

Download the preprocessed colbert document tensors data. The data is BZ2 compressed and each file is about 15GB compressed. 

<pre>
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-p1.bz2  -O sample-feed/colbert-passages-p1.bz2
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-p2.bz2  -O sample-feed/colbert-passages-p2.bz2
$ wget https://data.vespa.oath.cloud/colbert_data/colbert-passages-p3.bz2  -O sample-feed/colbert-passages-p3.bz2
</pre>

Feed all 8.8M passages 

<pre>
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /MSMARCO/sample-feed/passage-all-feed.jsonl --host localhost --port 8080'
</pre>

Update all 8.8M passages with colbert tensor data. Note that we stream through using *bunzip2* as the uncompressed representation
is large (JSON is not the best format for storing tensor data). 

<pre>
$ docker exec vespa bash -c 'bunzip2 -c /MSMARCO/sample-feed/colbert-passages-p*.bz2 | java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
     --host localhost --port 8080'
</pre>

## Ranking Evaluation using Ms Marco Passage Ranking *dev*


## Create your own submission 

# Appendix ColBERT example 
A toy example using 2 dimensions for the contextual term embedding for a input passage with 4 terms. 

The text is processed using a sub-word BERT tokenizer which maps the text to bert token_ids in a fixed vocabulary. The english BERT vocabulary 
has about 30K terms.  

<pre>
colbert_document_encoder("An colbert example passage") 
[
 [0.12 , 0.133 ],
 [0.39 , 0.34 ],
 [0.02 , 0.42 ],
 [0.77, 0.24 ]
]
</pre>

The above representation can be stored in Vespa document schema using Vespa [tensor fields](https://docs.vespa.ai/documentation/tensor-user-guide.html).
At run time we need to compute the same tensor representation of the query:

<pre>
colbert_query_encoder("passage ranking")
[
 [0.3 , 0.144 ],
 [0.34 , 0.32 ]
]
</pre>

The **MaxSim** score for our toy passage given the query *q*. The passage tensor representation can be pre-processed offline:

<pre>
score(q,d_id):
 d = get_tensor_data(d_id)
 inner_product = dot(q,d.transpose())
 [
   [0.055152, 0.16596 , 0.06648 , 0.26556 ],
   [0.08336 , 0.2414  , 0.1412  , 0.3386  ]
 ]
 max_sim = inner_product.max(1)
 [
    0.26556, 
    0.3386 
 ]
 score = sum(max_sim)
 score
 0.60416
 return score
</pre>


