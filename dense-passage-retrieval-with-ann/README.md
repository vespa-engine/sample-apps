<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

# Vespa sample application - Open-Domain Question Answering

Ever wondered what the answer to the question *Who came up with the name rock and roll* is?  Or maybe for *who is the prime minister of Norway?* 

In this Vespa.ai sample application we demonstrate how to build an 
open domain question answering **serving** system with state-of-the-art accuracy using Vespa.ai.

The system use Vespa.ai's support for [fast approximate nearest neighbor search](https://docs.vespa.ai/documentation/approximate-nn-hnsw.html) 
to perform dense embedding retrieval over Wikipedia for question answering using 
Facebook's [Dense Passage Retriever Model (DPR)](https://github.com/facebookresearch/DPR) described in the
[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) paper. Generally, representation learning 
is used to train a document (item) embedding model and question (user query) embedding model where documents which are relevant has a 
higher similarity in the embedding space than irrelevant documents. The DPR model uses [BERT](https://arxiv.org/abs/1810.04805)
 as the query and document embedding models, and
fine-tune the weights of the two embedding towers using the [Natural Questions](https://ai.google.com/research/NaturalQuestions)
train dataset which contains labeled data. The DPR paper achieves state-of-the-art accuracy on Open Domain Question Answering 
and the embedding based retriever outperforms traditional term based matching (BM25) significantly.

<figure>
<p align="center"><img width="90%" src="img/embedding_learning.png" /></p>
</figure> 

We take the DPR implementation, which is a set of python tools and models, and convert the models to Vespa.ai for online serving
while maintaining the same or better accuracy as reported in the DPR paper. 

* We index text passages from the English version of the Wikipedia along with their embedding representation produced by the DPR document 
encoder in a Vespa.ai instance. Representing DPR on Vespa.ai also allow researchers to experiment with different retrieval strategies. 
* We use the pre-trained DPR BERT based query embedding model from [Huggingface](https://huggingface.co/transformers/model_doc/dpr.html) 
which we export to [ONNX](https://onnx.ai/) format using Huggingface's [Transformer model export support](https://huggingface.co/transformers/serialization.html)
and we import this ONNX model to Vespa for serving so that given a textual query input we can convert it into the embedding representation at user time.
* The DPR query embedding representation is used as input to Vespa.ai's 
[fast approximate nearest neighbor search](https://docs.vespa.ai/documentation/approximate-nn-hnsw.html) which enables fast computing the 
top-k matching passages in the embedding space. 
* The top-k retrieved passages (Using multiple different retrieval strategies) 
are re-ranked using Vespa's support for multi-tier retrieval and ranking with another 
BERT based model which scores passages and computes the most likely answer span from the passages.
* We use Vespa's plugin support to implement a sub-word BERT tokenizer.
* We reproduce the state-of-the-art retrieval metrics and reader evaluation metrics as reported in the paper using the Open Domain variant of the Natural Questions.

Using Vespa.ai as serving engine for passage retrieval for question answering allows representing both sparse term based and dense embedding retrieval 
in the same schema, which also enables hybrid retrieval using a combination of the two approaches. With Vespa's support for running Transformer based
models like BERT via Vespa's ONNX run time support we are also able to deploy the DPR BERT query embedding encoder used for the dense embedding retrieval in 
the same serving system and also the DPR BERT based Reader component which re-scores the retrieved passages and predicts the 
best answer span.

## Question Answering System Architecture 

A typical architecture for Open Domain Question Answering consist of two main components:

* A **Retriever** Component which is capable of retrieving relevant passages for a question from a potentially very large document corpus
* A **Reader** Component which uses the retrieved context/passages to produce an answer to the question

<figure>
<p align="center"><img width="90%" src="img/vespa_passage_retrieval.png" /></p>
</figure> 

### DPR Retrieval Component
DPR uses a embedding based retrieval approach instead of term based (sparse, TF-IDF). To train the embedding representation they use 
the popular Two Tower Architecture using two BERT models for representation learning. The parameters of the query encoder and the document encoders
are trained given labeled data from Natural Questions Train set and the training process involves both showing relevant and non-relevant passages for
questions and the weights (parameters) of the two towers are adjusted so that relevant passages for a question has a higher similarity score than 
irrelevant passages (Which don't contain the answer to the question). 

Once the query and document representation models have been trained we can 
export them for serving. Documents are feed through the Document encoder which for each Wikipedia title and text passage produces the 768 dimensional embedding
representation. At serving time, the user question is encoded by the question encoder and the search for relevant passages is reduced to a nearest neighbor search problem. 

<figure>
<p align="center"><img width="90%" src="img/two-towers-embedding.png" /></p>
</figure>

### DPR Reader Component
The Reader component in DPR is a BERT model which is fine-tuned for question answering. The reader BERT model takes both the question, the wiki title and 
the wiki passage as input (With cross-attention between the question and the document). 

<figure>
<p align="center"><img width="90%" src="img/reader.png" /></p>
</figure> 

## Implementation the DPR architecture with Vespa.ai for Serving

We represent the Wikipedia passage text, title and the passage embedding vector 
in the same Vespa [document schema](src/main/application/schemas/wiki.sd). 
We also store the token ids from the BERT
tokenization as Vespa tensor fields. These token_ids fields are not used by the retriever component but by the reader. The
tensor field type in Vespa is always stored in memory for fast access during retrieval and ranking. The schema is defined below:

<pre>
schema wiki {

  document wiki {

    field title type string {
      indexing: summary | index
      index: enable-bm25
    }

    field title_token_ids type tensor<float>(d0[256]) {
        indexing: summary | attribute
    }

    field text type string {
      indexing: summary | index
      index: enable-bm25
    }

    field text_token_ids type tensor<float>(d0[256]) {
      indexing: summary |attribute
    }

    field id type long {
      indexing: summary | attribute
    }

    field text_embedding type tensor<float>(x[769]){
      indexing: attribute | index  
      attribute {
        distance-metric:euclidean
      }
      index {
        hnsw {
          max-links-per-node: 32
          neighbors-to-explore-at-insert: 500
        }
      }
    }
  }
  fieldset default {
    fields: title, text
  }
}
</pre> 

The above Vespa document schema allows retrieval using different strategies using the same scalable serving engine:

* **Sparse retrieval** Using traditional term based (High dimensional, sparse)
* **Dense retrieval** Using trained embedding representations of query and document (Low dimensional, dense) 
* **Hybrid** Using a combination of the above 

The schema defines 2 string fields which are indexed which enables fast and efficient term based retrieval, e.g using WeakAND (WAND). 
The *id* represents the Wikipedia passage id as assigned in the pre-computed dataset published by Facebook Research. 
The *text_embedding* tensor is a dense 769 dimensional tensor which represents the document (text and title) and
we enable [HNSW index for fast approximate nearest neighbor search](https://docs.vespa.ai/documentation/approximate-nn-hnsw.html). 

The dual query and document encoder of the DPR retrieval system uses the inner dot product between
the query tensor and the document tensor to represent the score. 
We transform the 768 dimensional inner product space
to euclidean space using an [euclidean transformation](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf) 
which adds one dimension so our representation becomes 
769 dimensional where we can use the euclidean distance metric when finding the nearest neighbors in embedding space.
The DPR implementation uses the same space transformation when using [Faiss with HNSW index](https://github.com/facebookresearch/faiss). 

 
## Vespa Retrieval & Ranking
We can express our retrieval strategies by 
* A Vespa [search api request](https://docs.vespa.ai/documentation/query-api.html) with a query 
specified using the [Vespa YQL query language](https://docs.vespa.ai/documentation/reference/query-language-reference.html) or we can build the query 
request programatically in a custom Searcher plugin written in Java. In this sample application we build the query retrieval trees using the latter approach. 
* A [ranking](https://docs.vespa.ai/documentation/ranking.html) specification which controls how we score documents retrieved by the query.  

Vespa assigns rank score using ranking expressions, configured in a ranking profile in the document schema. The rank profile
can also specify [multi-phased ranking](https://docs.vespa.ai/documentation/phased-ranking.html). Choosing rank profile is a run time 
request parameter (ranking.profile).

The ranking profile is also configured in the schema and our question answering ranking profile looks like this

<pre>
rank-profile openqa {
    constants {
      TOKEN_NONE: 0
      TOKEN_CLS:  101
      TOKEN_SEP:  102
    }

    # Find length of question input
    function question_length() {
      expression: sum(map(query(query_token_ids), f(a)(a > 0)))
    }

    #Find length of the title
    function title_length() {
      expression: sum(map(attribute(title_token_ids), f(a)(a > 0)))
    }

    #Find length of the text
    function text_length() {
      expression: sum(map(attribute(text_token_ids), f(a)(a > 0)))
    }

    # The attention mask has 1's for every token that is set
    function attention_mask() {
      expression: map(input_ids, f(a)(a > 0))
    }

    function input_ids() {
      expression {
        tensor<float>(d0[1],d1[380])(...)
      } 
    first-phase {
      expression: closeness(field, text_embedding)
    }
    second-phase {
      rerank-count: 10
      sum(onnxModel("files/reader.onnx", "output_2"))
    }
    summary-features {
      onnxModel(files_reader_onnx).output_0 #start score tensor 
      onnxModel(files_reader_onnx).output_1 #end score logits
      input_ids #The input sequence with special tokens (CLS/SEP) 
    }
}
</pre>
The *input_ids* function builds the input tensor to the ONNX model. The batch size is 1 and the max sequence length is 380 token _ids, including
special tokens like CLS and SEP. The function builds a tensor: [[CLS, question token_ids, SEP, title_token_ids, SEP, text_token_ids]] which is 
evaluated by the Reader ONNX model. The **summary-features** is a way to pass ranking features and tensors from the
content nodes to the java serving container.  


### Import Transformer models to Vespa.ai via ONNX

The DPR team has published the pre-trained checkpoints on [Huggingface](https://huggingface.co/models)' model repository :

* BERT based question encoder https://huggingface.co/facebook/dpr-question_encoder-single-nq-base
* BERT based document encoder https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base 
* BERT based reader https://huggingface.co/facebook/dpr-reader-single-nq-base 

We can export these Transformer models to 
 [ONNX](https://onnx.ai/) format using Huggingface's [Transformer model export support](https://huggingface.co/transformers/serialization.html):

* DPR Question encoder model [bin/query-model-export.py](bin/query-model-export.py)
* DPR Reader model [bin/model-export.py](bin/model-export.py)

In the following snippet we export the reader model to ONNX format and serialize it to *reader.onnx*.

<pre>
import onnx   
import transformers
import transformers.convert_graph_to_onnx as onnx_convert
from pathlib import Path 
from transformers import DPRReader, DPRReaderTokenizer
tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)
pipeline = transformers.Pipeline(model=model, tokenizer=tokenizer)
onnx_convert.convert_pytorch(pipeline, opset=11, output=Path("reader.onnx"), use_external_format=False)
</pre>

We can store these ONNX model into the [src/main/application/files/](src/main/application/files/) and we
can later then reference it by **onnxModel("files/reader.onnx", "output_2")** in the ranking expression as seen in the previous section. Vespa
takes care of distributing the model to the content nodes in the cluster.


### Vespa Container Middleware - putting it all together 
The application has 4 custom plugins: 

* A BERT Tokenizer component which does map text to BERT vocabulary token_ids. This is a shared component which both the custom Searcher
and Document processor uses. We store the token_ids of the text and the title in the document so we don't need to perform any run time tokenization.
* A custom Document Processor which does BERT tokenization during indexing.
[QADocumentProcessor.java](src/main/java/ai/vespa/processor/QADocumentProcessor.java)
* A custom Searcher which controls the Retrieval logic (Sparse, dense, hybrid) and uses the BERT Tokenizer to convert the question string to 
a sequence of token_ids.
[RetrieveModelSearcher.java](src/main/java/ai/vespa/searcher/RetrieveModelSearcher.java)
* A custom Searcher which reads the outputs of the reader model for the best ranking hit from the reader phase (Vespa second phase ranking) 
and maps the best matching answer span 
to an textual answer which is returned as the predicted answer. [QASearcher.java](src/main/java/ai/vespa/searcher/QASearcher.java)

## Experiments and Results
In the following section we describe the experiments we have performed with this setup, all experiments are done running queries
via the [Vespa query api](https://docs.vespa.ai/documentation/query-api) and
and checking the predicted answer against the golden reference answer. 

<pre>
def get_vespa_result(question, retriever_model):
  request_body = {
    'type': 'any',
    'query': question,
    'retriever': retriever_model
  }
  url = endpoint + '/search/'
  response = requests.post(url, json=request_body)
  return response.json()
</pre>

### Retriever Accuracy Summary 

The following table summarizes the retriever accuracy using the original 3,610 dev questions in the Natural Questions for
Open Domain Question Answering tasks ([NQ-open.dev.jsonl](https://github.com/google-research-datasets/natural-questions/blob/master/nq_open/NQ-open.dev.jsonl)).

We use Recall@K as the main evaluation metric for the retriever, the final top position passages are re-ranked using the full attention Reader.

The obvious goal of the retriever is to have the highest recall possible at the lowest possible position. The fewer passages we 
need to evaluate through the BERT reader the better the run time complexity and performance is. We evaluate three different retrieval strategies:

* **Dense** Using the DPR embeddings and Vespa's nearest neighbor search operator 
* **Sparse** Using the Vespa's [weakAnd(WAND)](https://docs.vespa.ai/documentation/using-wand-with-vespa.html) query operator and using BM25(title) + BM25(text) 
* **Hybrid** Using a linear combination of the above and using OR to combine the weakAnd and nearestNeighbor search operator.

| Retrieval Model                | Recall@1  | Recall@5 | Recall@10| Recall@20 |
|-------------------------------- |-----------|----------|----------|-----------|
| sparse (WAND bm25)              | 23.77     | 44.24    | 52.69    | 61.47     | 
| dense  (nearest neighbor)       | 46.37     | 68.53    | 75.07    | 80.36     |
| hybrid (WAND + nearest neighbor)| 40.61     | 69.25    | 75.96    | 80.44     |

The DPR paper reports Recall@20 79.4 so our results are in accordance with the reported results for the dense retrieval method. 

The following table summarizes the retriever accuracy using the 1,800 dev questions used in the 
[Efficient Open-Domain Question Answering challenge](https://efficientqa.github.io/) 
([NQ-open.efficientqa.dev.1.1.jsonl](https://github.com/google-research-datasets/natural-questions/blob/master/nq_open/NQ-open.efficientqa.dev.1.1.jsonl)).

| Retrieval Model                 | Recall@1  | Recall@5 | Recall@10| Recall@20 |
|---------------------------------|-----------|----------|----------|-----------|
| sparse (WAND bm25)              | 23.94     | 44.67    | 52.67    | 60.78     | 
| dense  (nearest neighbor)       | 41.78     | 66.11    | 73.28    | 77.94     |
| hybrid (WAND + nearest neighbor)| 36.94     | 66.94    | 74.28    | 78.06     |

To our knowledge there are no Retrieval accuracy reported yet for the *NQ-open.efficientqa.dev.1.1.jsonl*. 

### Reader Accuracy Summary 

We evaluate the Reader accuracy using the Exact Match (EM) metric. We report EM metrics for Reader re-ranking using top 5, top 10, and top 100 passages
from the retriever phase. The Exact Match metric measures the percentage of predictions that match any one of the ground truth answers **exactly**. 
To get an EM score of 1 for a query the answer prediction must match exactly the golden answer given in the dataset. For instance
for the question *when was the last moon landing* and the predicted answer *14 December 1972* it will not match the golden answers which 
are *14 December 1972 UTC* or *December 1972*.

**Original Natural Question dev set**
([NQ-open.dev.jsonl](https://github.com/google-research-datasets/natural-questions/blob/master/nq_open/NQ-open.dev.jsonl))

| Retrieval Model                 | EM(@5)   | EM (@10)| 
|---------------------------------|-----------|--------|
| sparse (WAND bm25               | 23.80     | 26.23  | 
| dense  (nearest neighbor)       | 39.34     | 40.58  | 
| hybrid (WAND + nearest neighbor)| 39.36     | 40.61  | 

**EfficientQA Natural Question dev set**
([NQ-open.efficientqa.dev.1.1.jsonl](https://github.com/google-research-datasets/natural-questions/blob/master/nq_open/NQ-open.efficientqa.dev.1.1.jsonl))

| Retrieval Model | EM(@5)   | EM (@10)| 
|-----------------|-----------|--------|
| sparse          | 21.22     | 24.72  | 
| dense           | 35.17     | 35.89  | 
| hybrid          | 35.22     | 35.94  | 

| Retrieval Model                 | EM(@5)   | EM (@10)| 
|---------------------------------|-----------|--------|
| sparse (WAND bm25               | 21.22     | 24.72  | 
| dense  (nearest neighbor)       | 35.17     | 35.89  | 
| hybrid (WAND + nearest neighbor)| 35.22     | 39.94  | 


## Reproducing this work  - Requirements for running this sample application:

* [Docker](https://www.docker.com/) installed and running  
* git client to checkout the sample application repository and DPR and Maven installed 
* Operating system: macOS or Linux, Architecture: x86_64
* Minimum **128GB** system memory 
* python3 and DPR dependencies, see [DPR repo](https://github.com/facebookresearch/DPR) 
 
See also [Vespa quick start guide](https://docs.vespa.ai/documentation/vespa-quick-start.html). 

## Checkout the sample-apps repository and install DPR requirements
<pre>
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/dense-passage-retrieval-with-ann; mkdir src/main/application/files
$ ./bin/model-export.py 
$ ./bin/query-model-export.py 
$ mv reader.onnx src/main/application/files/reader.onnx; mv question_encoder.onnx src/main/application/files/encoder.onnx
$ docker run --detach --name vespa --hostname vespa-container \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>

Wait for configuration service to start (Wait for the command below return a 200 OK):

<pre>
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

Build the application and deploy application:

<pre>
$ cd $VESPA_SAMPLE_APPS/dense-passage-retrieval-with-ann/
$ mvn package -U
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare \
  /vespa-sample-apps/dense-passage-retrieval-with-ann/target/applization.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>

## Setup DPR
<pre>
$ cd $VESPA_SAMPLE_APPS/dense-passage-retrieval-with-ann/
$ git clone --depth 1 https://github.com/facebookresearch/DPR.git
$ cd DPR; pip3 install .
</pre>

## Download the Wikipedia data and pre-computed embeddings

Thanks to [Facebook Research](https://opensource.fb.com/) for providing both the pre-tokenized Wikipedia text passages and the corresponding passage embeddings. 
Note that the data is large, the text passage representation (data.wikipedia_split) is 13G and the pre-computed embeddings is 62G.

To download the pre-generated Wikipedia snippets and the pre-computed passage embeddings use the DPR download utility: 
<pre>
$ python3 data/download_data.py  --resource data.wikipedia_split 
$ python3 data/download_data.py  --resource data.retriever_results.nq.single.wikipedia_passages
</pre>

## Join passage text and embedding to Vespa feed format

We join this data and create a Vespa feed file with one Vespa put document operation per line [Vespa json feed format](https://docs.vespa.ai/documentation/reference/document-json-format.html). 
The scripts reads the entire Wikipedia passage into memory and reads
one embedding file at a time and emit a join of the textual passage meta data with the precomputed DPR embedding. 
 
<pre>
$ cd $VESPA_SAMPLE_APPS/dense-passage-retrieval-with-ann/ 
$ ./bin/make-vespa-feed.py DPR/data/wikipedia_split/psgs_w100.tsv DPR/data/retriever_results/nq/single/wikipedia_passages_* > feed.jsonl
</pre>

Sample data emitted (newline formatted for readability):

<pre>
{
  "put": "id:wiki:wiki::3", 
  "fields": {
    "title": "Aaron", 
    "text": "his rod turn into a snake. Then he stretched out his rod in order to .. While Joshua went with Moses to the top,",
    "id": 3, 
    "text_embedding": {"values": [-0.23045846819877625,....]}
  }
}
</pre>

We are now ready to index the data in our Vespa installation. The feed file is 273G uncompressed. 

## Feed data to Vespa

We feed the documents using the [Vespa http feeder client](https://docs.vespa.ai/documentation/vespa-http-client.html):
<pre>
$ java -jar vespa-http-client-jar-with-dependencies.jar --file feed.jsonl --endpoint http://your-vespa-instance-hostname:8080 
</pre>
We can obtain the feeding client jar from the docker image by :
<pre>
docker cp vespa:/opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar . 
</pre>

Loading the data to Vespa using a single content node instance with 36 vcpu's takes about 5 hours (21M passages , 1350 puts/s sustained, with visibility-delay 1.0 seconds
and real time indexing). Note that indexing both build inverted indexes for efficient sparse term based retrieval and HNSW graph for fast efficient dense embedding retrieval. 

## Experiments 

### Retriever experiments 
To run all questions from the Natural Questions (NQ) dev split using the three different retrieval strategies 
do: 
<pre>
$ cd $VESPA_SAMPLE_APPS/dense-passage-retrieval-with-ann/
$ wget https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl
$ ./bin/evaluate_em.py NQ-open.dev.jsonl dense http://your-vespa-instance-hostname:8080
$ ./bin/evaluate_em.py NQ-open.dev.jsonl sparse http://your-vespa-instance-hostname:8080
$ ./bin/evaluate_em.py NQ-open.dev.jsonl hybrid http://your-vespa-instance-hostname:8080
</pre>

You can also run queries in your browser, 
just head to [who sings ain't nothing but a good time](http://your-vespa-instance-hostname:8080/search/?query=who%20sings%20ain%27t%20nothing%20but%20a%20good%20time)

