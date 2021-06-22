<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

# MS Marco Passage Ranking using Transformers 

This sample appplication demonstrates how to efficiently represent three different ways of applying pretrained Transformer
models for text ranking in Vespa. The three methods are described in detail in 
these blog posts:

* [Pretrained language models for search - part 1 ](https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-1/)
* [Pretrained language models for search - part 2 ](https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-2/)
* [Pretrained language models for search - part 3 ](https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-3/)
* [Pretrained language models for search - part 4 ](https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-3/)

With this sample application you can reproduce our 
[MS Marco Passage Ranking Leaderboard](https://microsoft.github.io/MSMARCO-Passage-Ranking-Submissions/leaderboard/) 
submission which currently ranks #15, above many huge ensemble models using large Transformer models (e.g T5 using 3B parameters).

![MS Marco Leaderboard](img/leaderboard.png)


## Transformers for Ranking 
![Colbert overview](img/colbert_illustration.png)

*Illustration from [ColBERT paper](https://arxiv.org/abs/2004.12832)*

## Sample application 

In this sample application we demonstrate:

- Simple single stage retrieval accelerated by the [WAND](https://docs.vespa.ai/en/using-wand-with-vespa.html) 
dynamic pruning algorithm with [BM25](https://docs.vespa.ai/en/reference/bm25.html) ranking.  
- Dense (vector) search retrieval to replace sparse (BM25) for efficient candidate retrieval using Vespa's support for fast
[approximate nearest neighbor search](https://docs.vespa.ai/en/approximate-nn-hnsw.html).
 We demonstrate how to also embed the query encoder model which encodes the query text to vector representation. This
model is a representation model. Illustrated in figure **a**. 
- Re-ranking using the [Late contextual interaction over BERT (ColBERT)](https://arxiv.org/abs/2004.12832) model where we
also embed the ColBERT query encoder and we use Vespa tensor expressions to calculate the *MaxSim*.  Illustrated in figure **d**. 
- Re-ranking using all to all cross attention. Illustrated in figure **c**.
- [Multiphase retrieval and ranking](https://docs.vespa.ai/en/phased-ranking.html).
- Custom [searcher plugins](https://docs.vespa.ai/en/searcher-development.html), 
[document processors](https://docs.vespa.ai/en/document-processing.html) .


In this work we use three pre-trained Transformer models fine-tuned for MS Marco passage ranking from [Huggingface 🤗](https://huggingface.co/)

All three models are based on [MiniLM](https://arxiv.org/abs/2002.10957) which
is a distilled BERT model which can be used a drop in replacement for BERT. 
It uses the same tokenization procedure and vocabulary as BERT base.
It performs roughly with the same accuracy as the more known big brother *bert-base-uncased* but with
 less parameters which lowers the compute complexiy. 
 
 The original MiniLM has 12 layers, we use 6 layer versions with about 22.7M trainable parameters.  

**Transformer Models**
- Dense retrieval using bi-encoder [🤗 sentence-transformers/msmarco-MiniLM-L-6-v3](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-6-v3)
The original model uses mean pooling over the last layer of the MiniLM model, we also add a L2 normalization to normalize vectors to unit length (1) so that
we can use innerproduct distance metric instead of angular distance metric. This saves computations during the approximate nearest neighbor search. 
- Contextualized late interaction [🤗 vespa-engine/col-minilm](https://huggingface.co/vespa-engine/col-minilm)
- Cross all to all encoder [🤗 cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) 
 
We include code to export these three models to [ONNX](https://docs.vespa.ai/en/onnx.html) format for efficient serving in Vespa.ai. For all
three we use [quantized](https://www.onnxruntime.ai/docs/how-to/quantization.html) versions to speed up inference on CPU.  

We also publish the ONNX files for raw import if you don't want to export the models yourself. 

## MS Marco Passage Ranking Evaluation 

The official ranking evaluation metric on MS Marco passage leaderboard is [MRR@10](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 
and below are the results of our 2 passage ranking submissions on the **eval** and **dev** query set. 

The MS Marco Passage ranking leaderboard is sorted by the score on the *eval* set. 
See [MSMarco Passage Ranking Leaderboard](https://microsoft.github.io/msmarco/)
 
For reference we also include 
the official baseline using [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) and 
a tuned BM25 using *Apache Lucene 8* which powers text ranking search for engines like *Apache Solr* and *Elasticsearch*. 

| Model                                     | Eval  | Dev   |
|-------------------------------------------|-------|-------|
| BM25 (Official baseline)                  | 0.165 | 0.167 |
| BM25 (Lucene8, tuned)                     | 0.190 | 0.187 |
| **BM25 => ColBERT**                       | 0.347 | 0.354 |
| **dense => ColBERT => Cross all to all**  | 0.393 | 0.403 |

Methods in bold are end to end represented on Vespa.

The accuracy as measured by MRR@10 for the re-ranking runs depends on the re-ranking depths (how many documents each phase re-ranks). 

## Vespa document schema 

Vectors or generally tensors are first-class citizens in the Vespa document model and we use the following 
Vespa [passage](src/main/application/schemas/passage.sd) document schema with 5 fields:

<pre>
schema passage {

  document passage {

    field text type string {
      indexing: summary | index
      index: enable-bm25
    }
    
    field text_token_ids type tensor&lt;float&gt;(d0[128])  {
      indexing: summary | attribute
    }
    
    field dt type tensor&lt;bfloat16&gt;(dt{}, x[32]){
      indexing: summary | attribute
    }
    
    field mini_document_embedding type tensor&lt;float&gt;(d0[384]) {
      indexing: attribute | index
      attribute {
        distance-metric: innerproduct
      }
      index {
        hnsw {
          max-links-per-node: 32
          neighbors-to-explore-at-insert: 500
        }
      }
    }
    field id type int {
      indexing: summary |attribute
    }
  }
} 
</pre> 

The *text* field contains the original passage text and the *dt* tensor field stores the contextual term embedding from the col-MiniLM model. *dt* is an example of a mixed tensor
which mixes sparse ("dt") and dense "x". Using sparse allows us to store several term embeddings without a fixed dense dimensionality as the text is of variable length.

We use 32 dimensions for the contextualized term embedding and this is denoted by <em>x[32]</em>. We use bfloat16 (2 bytes) per tensor cell to reduce memory footprint.

 
See [Vespa Tensor Guide](https://docs.vespa.ai/en/tensor-user-guide.html) for an introduction to Vespa tensors. The *id* field is the passage id from the 
dataset. This id is used to evaluate the ranking accuracy. We also define a *id* document summary which avoids returning the text and the potential large *dt* tensor field. 

The *text_token_ids* contains the BERT token vocabulary ids and is only used by the final cross all to all interaction re-ranking model. 
Storing the token ids in the passage document avoids tokenization at query serving time 
and we can read the tensor from memory during ranking. 

The *mini_document_embedding* field is the dense vector produced by the sentence encoder model. We enable HNSW indexing for efficient fast 
approximate nearest neighbor search and we use innerproduct distance metric (query and document vectors normalized to unit length). 


## Retrieval and Ranking 
We define several ranking profile in our *passage* document schema. See [Vespa Ranking Documentation](https://docs.vespa.ai/en/ranking.html)
for an overview of how to represent ranking in Vespa.

The baseline ranking model is using [bm25](https://docs.vespa.ai/en/reference/bm25.html). BM25 could be
a good baseline model for many domains where there are no prior training data available. Be it by explicit relevancy judgements 
like in MS Marco or implicit through user feedback. When there is enough training data available it's possible to train 
a representation model which significantly outperforms BM25 in-domain. 

Below is the the *bm25* ranking profile, defined in the passage document schema:

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

See [Vespa performance and sizing guide](https://docs.vespa.ai/en/performance/sizing-search.html).

 
We define the *colBERT* reranker with a Vespa ranking profile which uses [phased ranking](https://docs.vespa.ai/en/phased-ranking.html). 
The first phase is inherited from the *bm25* ranking profile. The way this work is that the top K documents as scored by 
the bm25(text) ranking feature is re-ranked using the ColBERT MaxSim operator. The re-ranking count or re-ranking depth is configurable 
 and can also be overridden at query time.
 
The ColBERT MaxSim operator is expressed using Vespa's
[tensor expression language](https://docs.vespa.ai/en/reference/ranking-expressions.html#tensor-functions). 

 
<pre> 
rank-profile bm25-colbert inherits bm25 {
  num-threads-per-search: 6
  second-phase {
    rerank-count: 1000
    expression {
      sum(
        reduce(
          sum(
            query(qt) * cell_cast(attribute(dt),float), x
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
The **cell_cast** is used to cast from bfloat16 format in memory to float for faster computation on cpu. 

The query tensor type is defined in the application package in
[src/main/application/search/query-profiles/types/root.xml](src/main/application/search/query-profiles/types/root.xml):
<pre>
&lt;field name="ranking.features.query(qt)" type="tensor&lt;float&gt;(qt{},x[32])"/&gt;
</pre> 

There are several ranking profiles defined in the passage schema and the best performing model is the following


<pre>
  rank-profile dense-colbert-mini-lm {
    num-threads-per-search: 12

    function input_ids() {
        expression: tokenInputIds(128, query(query_token_ids), attribute(text_token_ids))
    }

    function token_type_ids() {
      expression: tokenTypeIds(128, query(query_token_ids), attribute(text_token_ids))
    }

    function attention_mask() {
      expression: tokenAttentionMask(128, query(query_token_ids), attribute(text_token_ids))
    }

    #Max score is 32 * 1.0
    function maxSimNormalized() {
      expression {
        sum(
          reduce(
            sum(
              query(qt) * attribute(dt), x
            ),
            max, dt
          ),
          qt
        )/32.0
       }
    }
    function dense() {
      expression: closeness(field,mini_document_embedding)
    }

    function crossModel() {
      expression: onnx(minilmranker){d0:0,d1:0}
    }

    first-phase {
        expression: maxSimNormalized()
    }

    second-phase {
      rerank-count: 24
      expression: 0.2*crossModel() + 1.1*maxSimNormalized() + 0.8*dense()
    }
  }
</pre>

This ranking model uses the dense nearest neighbor search to compute the top 1K documents, which are
re-ranked using the late interaction model and finally the top k from the late interaction model
is re-ranked once more using the cross all to all interaction model. We also use the ranking
signals computed by the two previous stages using a linear combination of the 3 signals. The 
*dense()* and *maxSimNormalized* functions are not re-evaluated in the second phase. Also note that
that these re-ranking steps are performed per node without crossing the network.  


### Query encoders 

We represent both the ColBERT and the dense query encoder in Vespa 
using Vespa's support for inference and [ranking with ONNX](https://docs.vespa.ai/en/onnx.html) models. 
In this case we don't rank any documents directly with the ONNX model but we use the support to make a single pass through them
to obtain the embedding vector for nearest neighbor search and the ColBERT query tensor representation for re-ranking. 
 
We plug the model into the Vespa serving architecture 
using a *query* document type which is empty as it's only used to represent the query encoder models

<pre> 
schema query {
  document query {
  }
  onnx-model colbert_encoder {
    file: files/vespa-colMiniLM-L-6-quantized.onnx
    input input_ids: query(input_ids)
    input attention_mask: query(attention_mask)
    output contextual:contextual 
  }
    
  onnx-model query_embedding {
    file: files/sentence-msmarco-MiniLM-L-6-v3-quantized.onnx
    input input_ids: query(input_ids)
    input attention_mask: query(attention_mask)
    output output_0: embedding
  }
}
</pre>


## Scaling and Serving Performance

See [Scaling and performance evaluation of ColBERT on Vespa.ai](colbert-performance-scaling.md)


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
* Java 11, Maven and python3
* zstd: `brew install zstd`
* Operating system: macOS or Linux, Architecture: x86_64

First, we clone the sample apps :

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/msmarco-ranking
</pre>

Install python dependencies. There are no run time python dependencies in Vespa, but this sample app uses:

<pre data-test="exec">
$ pip3 install torch numpy ir_datasets requests tqdm transformers onnx onnxruntime
</pre>

Download ONNX models which have been exported by us and made available for this sample app. 

We place these models into the files directory of the Vespa application package:

<pre data-test="exec">
$ mkdir -p src/main/application/files/
</pre>

In total we use three ONNX models, please follow the instructions to download them and 
place them in the files directory with the correct file name. The mode files are referenced
in the *passage* and *query* vespa doucment schemas.

## The sentence encoder model (bi-encoder)

This model is used for dense retrieval using HNSW indexing. The model uses mean pooling 
and we add an L2 normalization on top so we can use innerproduct distance metric instead
of angular. The mean pooling and normalization is part of the onnx model generation. 

<pre data-test="exec">
$ curl -L -o src/main/application/files/sentence-msmarco-MiniLM-L-6-v3-quantized.onnx \
    https://data.vespa.oath.cloud/onnx_models/sentence-msmarco-MiniLM-L-6-v3-quantized.onnx
</pre>

## The cross encoder model (classification)

This is the final re-ranking stage using full all to all
interaction between the query and the passage with a classification head on top of the CLS token
embedding:

<pre data-test="exec">
$ curl -L -o src/main/application/files/ms-marco-MiniLM-L-6-v2-quantized.onnx \
    https://data.vespa.oath.cloud/onnx_models/ms-marco-MiniLM-L-6-v2-quantized.onnx
</pre>

## The late interaction model 
This is the ColMiniLM model which uses late contextualized interaction and Vespa MaxSim tensor expression:

<pre data-test="exec">
$ curl -L -o src/main/application/files/vespa-colMiniLM-L-6-quantized.onnx \
    https://data.vespa.oath.cloud/onnx_models/vespa-colMiniLM-L-6-quantized.onnx
</pre>

To see how we exported these models to onnx format see the [model export notebook](src/main/python/model-exporting.ipynb). We have
also released the models so you don't need to export them. 

Once we have downloaded the models, 
we use maven to create the
[Vespa application package](https://docs.vespa.ai/en/reference/application-packages-reference.html).

<pre data-test="exec">
$ mvn clean package -U
</pre>

If you run into issues running mvn package please check  mvn -v and that the Java version is 11. 
Now, we are ready to start the vespeengine/vespa docker container - pull the latest version and run it by

<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>

Wait for configuration service to start, the command below should return a 200 OK:

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:19071/ApplicationStatus
</pre>

Deploy the application package:

<pre data-test="exec" data-test-assert-contains="prepared and activated.">
$ curl --max-time 300 --header Content-Type:application/zip --data-binary @target/application.zip \
  localhost:19071/application/v2/tenant/default/prepareandactivate
</pre>

The vespa configuration will translate the high level application package and distribute 
models and configuration to all the vespa sub-services, potentially running over thousands of machines. 

Now, wait for the application to start, the command below should return a 200 OK

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

## Feeding Sample Data 

Feed the sample documents using the [Vespa http feeder client](https://docs.vespa.ai/en/vespa-http-client.html):

<pre data-test="exec">
$ curl -L -o vespa-http-client-jar-with-dependencies.jar \
    https://search.maven.org/classic/remotecontent?filepath=com/yahoo/vespa/vespa-http-client/7.391.28/vespa-http-client-7.391.28-jar-with-dependencies.jar
</pre>

Download the sample data:

<pre data-test="exec">
$ curl -L -o sample-feed/colmini-passage-feed-sample.jsonl.zst \
    https://data.vespa.oath.cloud/colbert_data/colmini-passage-feed-sample.jsonl.zst
</pre>

Feed the data :

<pre data-test="exec">
$ zstdcat sample-feed/colmini-passage-feed-sample.jsonl.zst | \
    java -jar vespa-http-client-jar-with-dependencies.jar \
     --endpoint http://localhost:8080
</pre>

Feed the query document which enables the query encoding 

<pre data-test="exec">
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --file sample-feed/sample_query_feed.jsonl --endpoint http://localhost:8080
</pre>

Now all the data is indexed and one can play around with the search interface. Note, only searching 1K passages.

For example do a query for *what was the Manhattan Project*: 

In this case using dense retrieval and colbert re-ranking 

<pre data-test="exec">
$ cat sample-feed/query.json
{
  "query": "what was the manhattan project?",
  "hits": 5,
  "queryProfile": "dense-colbert"
}
</pre>

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s -H Content-Type:application/json --data @sample-feed/query.json \
    http://localhost:8080/search/ |python3 -m json.tool
</pre>

It is also possible to view the result in a browser using HTTP GET

![Colbert example](img/colbert_sample_query.png)

### Retrieval and (re)ranking examples

A set of predefined methods for doing different retrieval and re-ranking methods is 
configured using [query profiles](https://docs.vespa.ai/en/query-profiles.html) 

See query profile definitions in [src/main/application/search/query-profiles](src/main/application/search/query-profiles)

Examples of retrieval and ranking methods demonstrated with this sample application:

| **Retrieval method** | **Ranking model**           | **Query example link**                                                                                           |
|------------------|----------------------|--------------------------------------------------------------------------------------------------------------|
| sparse (wand)    | bm25                          | [sparse-bm25](http://localhost:8080/search/?query=what+was+the+Manhattan%20Project&queryProfile=sparse-bm25) |
| sparse (wand)    | bm25 => col-MiniLM            | [sparse-colbert](http://localhost:8080/search/?query=what+was+the+Manhattan%20Project&queryProfile=sparse-colbert) |
| dense (ann)      | dense 1K                      | [dense](http://localhost:8080/search/?query=what+was+the+Manhattan%20Project&queryProfile=dense) |
| dense (ann)      | dense 10                      | [dense](http://localhost:8080/search/?query=what+was+the+Manhattan%20Project&queryProfile=dense&ann.hits=10) |
| dense (ann)      | dense => col-MiniLM           | [dense-colbert](http://localhost:8080/search/?query=what+was+the+Manhattan%20Project&queryProfile=dense-colbert) |
| dense (ann)      | dense => col-MiniLM => MiniLM | [dense-colbert-cross](http://localhost:8080/search/?query=what+was+the+Manhattan%20Project&queryProfile=dense-colbert-cross) |

It's possible to control approximate nearest neighbor search (ANN) and WAND parameters by  

* *&ann.hits=x* sets the targetNumber of hits that should be retrieved by the dense retriever and exposed to the first-phase ranking expression
* *&wand.hits=x* sets the targetNumber of hits that should be retrieved by the sparse wand retriever and exposed to the first-phase ranking expression


## Shutdown and remove the Docker container:

<pre data-test="after">
$ docker rm -f vespa
</pre>

## Full Evaluation

Full evaluation requires the content node to run on an instance with at least 128GB of memory. For optimal serving performance
a cpu with avx512 and VNNI instruction support is recommended. In our experiments we have used 2 x Xeon Gold 6263CY 2.60GHz (48, 96 threads).
 
### Download all passages 

Feed query document 

<pre>
$ java -jar vespa-http-client-jar-with-dependencies.jar \
    --file sample-feed/sample_query_feed.jsonl --endpoint http://localhost:8080
</pre>

### Download pre-processed ColMiniLM document representation 
Download the preprocessed document feed data which includes colbert term document embeddings and sentence transformer
embedding. The data is compressed using [ZSTD](https://facebook.github.io/zstd/): 

Each file is just below 20GB of data 

<pre>
$ for i in 1 2 3; do curl -L -o sample-feed/colmini-passage-feed-$i.jsonl.zst \
  https://data.vespa.oath.cloud/colbert_data/colmini-passage-feed-$i.jsonl.zst; done
</pre>

Note that we stream through the data using *zstdcat* as the uncompressed representation
is large (170GB). 

<pre>
$ zstdcat sample-feed/colmini-passage-feed-*.zst | \
    java -jar vespa-http-client-jar-with-dependencies.jar \
    --endpoint http://localhost:8080
</pre>

Indexing everything on a single node using real time indexing takes a few hours, depending on HW configuration (1000 puts/s). 


### Ranking Evaluation using Ms Marco Passage Ranking 

With the [evaluate_passage_run.py](src/main/python/evaluate_passage_run.py) we can run retrieval and ranking using the methods
demonstrated in this sample application


**BM25(WAND) Single phase sparse retrieval**
<pre>
$ ./src/main/python/evaluate_passage_run.py --query_split dev --retriever sparse \
  --rank_profile bm25 --wand_hits 100 --hits 10 --run_file run.dev.txt --endpoint \
  http://localhost:8080/search/
</pre>

**BM25(WAND) + ColMiniLM re-ranking**
<pre>
$ ./src/main/python/evaluate_passage_run.py --query_split dev --retriever sparse \
  --rank_profile bm25-colbert --wand_hits 1000 --rerank_hits 1000 --hits 10 --run_file run.dev.txt --endpoint \
  http://localhost:8080/search/
</pre>

**dense(ANN) Single phase dense retrieval**
<pre>
$ ./src/main/python/evaluate_passage_run.py --query_split dev --retriever dense \
  --rank_profile dense --ann_hits 10 --hits 10 --run_file run.dev.txt --endpoint \
  http://localhost:8080/search/
</pre>

**dense(ANN) + ColMiniLM re-ranking**
<pre>
$ ./src/main/python/evaluate_passage_run.py --query_split dev --retriever dense \
  --rank_profile dense-colbert --ann_hits 1000 --rerank_hits 1000 --hits 10 --run_file run.dev.txt --endpoint \
  http://localhost:8080/search/
</pre>

**dense(ANN) + ColMiniLM re-ranking + CrossMiniLm**
<pre>
$ ./src/main/python/evaluate_passage_run.py --query_split dev --retriever dense \
  --rank_profile dense-colbert-mini-lm --ann_hits 1000 --rerank_hits 24 --hits 10 --run_file run.dev.txt --endpoint \
  http://localhost:8080/search/
</pre>

To evaluate ranking accuracy download the official MS Marco evaluation script

<pre>
$ curl -L -o msmarco_eval.py https://raw.githubusercontent.com/spacemanidol/MSMARCO/master/Ranking/Baselines/msmarco_eval.py
</pre>

Generate the dev qrels (query relevancy labels) file using the *ir_datasets*. 

<pre>
$ ./src/main/python/dump_passage_dev_qrels.py
</pre>

Above will write a **qrels.dev.small.tsv** file to the current directory, now we can evaluate using 
the **run.dev.txt** file created by any of the evaluate_passage_run.py runs listed above 

<pre>
$ python3 msmarco_eval.py qrels.dev.small.tsv run.dev.txt
#####################
MRR @10: 0.xx
QueriesRanked: 6980
#####################
</pre>


# Appendix ColBERT 

Model training and offline text to tensor processing

* The *ColBERT* model is trained the instructions from the [ColBERT repository](https://github.com/stanford-futuredata/ColBERT) 
using the MS Marco Passage training set. The *bert-base-uncased* is replaced with the *MiniLM-L6*. 
We use cosine similarity (innerproduct as the vectors are unit length normalized).
* The dimensionality of the token tensor is reduced from 384 (hidden dim) to 32 dimensions by a linear layer
* GPU powered indexing routine in the mentioned *ColBERT repository* to obtain the document tensor representation


-------------------

Further reading:
* https://docs.vespa.ai/en/vespa-quick-start.html
* https://docs.vespa.ai/en/getting-started.html
