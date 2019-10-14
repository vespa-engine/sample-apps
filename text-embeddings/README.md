<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - Semantic Search using Text Embedddings 

This sample application contains code, document schema and dependencies for running https://docs.vespa.ai/documentation/semantic-search-with-word-embeddings.html 

**Tensor definitions**
See the [Tensor Intro in the Vespa](https://docs.vespa.ai/documentation/tensor-intro.html). In this sample application we defined two tensor type fields in the [Vespa document schema](src/main/application/searchdefinitions/passage.sd):
<pre>
field passage_embedding type tensor<float>(x[512]) {
  indexing: attribute
}
field sentence_embeddings type tensor<float>(s{},x[512]) {
  indexing: attribute
}
</pre>


**Checkout the sample-apps repository**
<pre>
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git; cd sample-apps/text-embeddings
</pre>

**Build a docker image (See [Dockerfile](Dockerfile) for details)**
The image builds on the [vespaengine/vespa docker image (latest)](https://hub.docker.com/r/vespaengine/vespa/tags) and installs python3 and the python dependencies tensorflow/tensorflow-hub and Natural Language Toolkit (nltk)**
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

After the above we have two new files in the working directory: _queries.txt_ and _vespa_feed.json_. The _queries.txt_ file contains queries which have at least one relevant passage and in our head 100 sample that equals to 97 queries. The total number of passages is 796 passages.

**Feed Vespa json** 
We feed the documents using the [Vespa http feeder client](https://docs.vespa.ai/documentation/vespa-http-client.html):
<pre>
$ java -jar $VESPA_HOME/lib/jars/vespa-http-client-jar-with-dependencies.jar --file vespa_feed.json --endpoint http://localhost:8080 
</pre>

**Run query evaluation of 6 different models/rank profiles**
The evaluation script runs all queries read from _stdin_ and for each query it executes n rank-profiles and finally it computes the [mean reciprocal rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) _MRR@10_ metric per rank profile for all queries:

<pre>
$ cat queries.txt |./text-embeddings/bin/evaluation.py 2> /dev/null
</pre>

Which should output :
<pre>
Rank Profile 'random' MRR@10 Result: 0.3164 
Rank Profile 'bm25' MRR@10 Result: 0.4893 
Rank Profile 'nativeRank' MRR@10 Result: 0.4924 
Rank Profile 'passage-similarity' MRR@10 Result: 0.4323 
Rank Profile 'sentence-similarity' MRR@10 Result: 0.4557 
Rank Profile 'native-similarity' MRR@10 Result: 0.4901 
</pre>

Please note that the lower bound for MRR@10 in this case where we re-rank 10 passages per query is is 1/10 


