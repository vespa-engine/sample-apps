<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - Semantic Search using Text Embedddings 

This sample application contains code, document schema and dependencies for running examples from https://docs.vespa.ai/documentation/semantic-search-with-word-embeddings.html 
where we use [Google's Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2) to encode text passages and text sentences into dense 512 dimensional tensors 
.  We index both the textual representation and the tensors fields in the same Vespa document schema and propose several different ranking models to rank passages of text from the 
[MS Marco](http://www.msmarco.org/)Q & A V1.1 dataset.

**Requirements:**

* [Docker](https://www.docker.com/) installed and running  
* git client to checkout the sample application repository

See also [Vespa quick start guide](https://docs.vespa.ai/documentation/vespa-quick-start.html). This setup is slightly different then the official quicks start guide.

**Tensor definitions**

See the [Tensor Intro](https://docs.vespa.ai/documentation/tensor-intro.html) in the Vespa documentation for an introduction to tensors and tensor operations in Vespa.
 In this sample application we define two tensor type fields in the use case specific [Vespa document schema](src/main/application/searchdefinitions/passage.sd):

<pre>
#Dense first order tensor with 512 dimensions
field passage_embedding type tensor<float>(x[512]) {
  indexing: attribute
}
#A second order tensor (matrix) which maps a sparse sentence id to the corresponding dense 
#sentence embedding of 512 dimensions
field sentence_embeddings type tensor<float>(s{},x[512]) {
  indexing: attribute
}
</pre>

To install and play around with the ranking models using the data set one needs to:

**Checkout the sample-apps repository**

This step requires that you have a working git client:
<pre>
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git; cd sample-apps/text-embeddings
</pre>

**Build a docker image (See [Dockerfile](Dockerfile) for details)**

The image builds on the [vespaengine/vespa docker image (latest)](https://hub.docker.com/r/vespaengine/vespa/tags) and installs python3 and the python dependencies tensorflow/tensorflow-hub and Natural Language Toolkit (nltk)
<pre>
$ docker build . --tag vespa_text_embeddings:1.0
</pre>
**Run the docker container built in the previous step and enter the running docker container**

<pre>
$ docker run --detach --name vespa_text_embeddings  --hostname vespa-container --privileged vespa_text_embeddings:1.0
$ docker exec -it vespa_text_embeddings  bash 
</pre>
**Deploy the document schema and configuration - this will start Vespa services**

<pre>
$ vespa-deploy prepare text-embeddings/src/main/application/ && vespa-deploy activate
</pre>

**Download the MS MARCO Q A V1.1 dataset and convert format to Vespa (Sample of 100 queries)**

Converting the [MS MARCO](http://www.msmarco.org/) Question Answering V1.1 dataset json format to Vespa also includes using the [Google Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2) to encode text passages and sentences to their tensor represenation. 

Downloading and converting
the sample 100 queries is not instant so grab a cup of tea while Tensorflow downloads the model from the Tensorflow Hub.
<pre>
$ ./text-embeddings/bin/download.sh
$ head -100 dev_v1.1.json |./text-embeddings/bin/convert-to-vespa.py  2> /dev/null
</pre>

After the above we have two new files in the working directory: _queries.txt_ and _vespa_feed.json_. The _queries.txt_ file contains queries which have at least one relevant passage and in our head 100 sample that equals to 97 queries. The total number of passages is 796.

**Feed Vespa json** 

We feed the documents using the [Vespa http feeder client](https://docs.vespa.ai/documentation/vespa-http-client.html):
<pre>
$ java -jar $VESPA_HOME/lib/jars/vespa-http-client-jar-with-dependencies.jar --file vespa_feed.json --endpoint http://localhost:8080 
</pre>

**Run query evaluation of 6 different models/rank profiles**

The evaluation script runs all queries read from _stdin_ and for each query it executes n rank-profiles and finally it computes the [mean reciprocal rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) _MRR@10_ metric per rank profile for all queries. The evaluations script uses the [Vespa search api](https://docs.vespa.ai/documentation/search-api.html) and each query limits the recall to the set of passages 
associated with the query (relevant or not) and the evaluation measures how well we are able to rank the set of passages on a per rank profile basis. The search request looks like this where 
the recall parameter limits the recall to only match against passages which are associated with the query id. We use _type_ any and include a query term which is always true for the cases
where there is no textual overlap between the user query and the text passage. 

<pre>
def handle_query(query,queryid,rank_profile):
  embedding = session.run(embeddings,feed_dict={text_sentences:[query]})[0].tolist()
  json_request = {
    "query": "(query_id:>0 %s)" % query,
    "type": "any",
    "hits": 10,
    "recall": "+query_id:%s" % queryid,
    "timeout":20, 
    "ranking.softtimeout.enable":"false",
    "ranking.features.query(tensor)": embedding,
    "ranking.profile": rank_profile
  }
  r = requests.post('http://localhost:8080/search/', json=json_request)
  response = r.json()
  if response["root"]["fields"]["totalCount"] == 0:
    return [0]
  selected = []
  for hit in response["root"]["children"]:
    selected.append(hit["fields"]["is_selected"])
  return selected
</pre> 

Running the _evaluation.py_ script:

<pre>
$ cat queries.txt |./text-embeddings/bin/evaluation.py 2> /dev/null
</pre>

Which should produce output like this:
<pre>
Rank Profile 'random' MRR@10 Result: 0.3010 
Rank Profile 'passage-semantic-similarity' MRR@10 Result: 0.4323 
Rank Profile 'max-sentence-semantic-similarity' MRR@10 Result: 0.4456 
Rank Profile 'bm25' MRR@10 Result: 0.4893 
Rank Profile 'nativeRank' MRR@10 Result: 0.4924 
Rank Profile 'nativeRank-and-max-sentence-linear' MRR@10 Result: 0.4901 
</pre>

Please note that the lower bound for MRR@10 in this case where we re-rank 10 passages per query is is 1/10.  
