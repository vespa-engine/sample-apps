<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - Semantic Search using Text Embedddings 

This sample application contains code and dependencies for running https://docs.vespa.ai/documentation/semantic-search-with-word-embeddings.html 

**Checkout the sample-apps repository**
<pre>
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git; cd sample-apps/text-embeddings
</pre>

** Build a docker image (See [Dockerfile] for details ) which installs python3, tensorflow/tensorflow-hub and Natural Language Toolkit (nltk)
<pre>
$ docker build . --tag text_embeddings:1.0
</pre>
**Run the docker container built in the previous step and enter the running docker container**
<pre>
$ docker run --detach --name vespa_text_embeddings  --hostname vespa-container --privileged text_embeddings:1.0
$ docker exec -it vespa_text_embeddings  bash 
</pre>
**Deploy the document schema and configuration - this will start Vespa services**
<pre>
$ vespa-deploy prepare text-embeddings/src/main/application/ && vespa-deploy activate
</pre>

**Download the MS MARCO dataset and convert MS Marco format to Vespa (Sample of 100 queries)**
Converting the ms marco json to vespa also includes using the Universal Sentence Encoder which needs to be downloaded. Downloading and converting
the sample 100 queries is not instant so grab a coffe while Tensorflow downloads the model from the Tensorflow Hub.
<pre>
$ ./text-embeddings/bin/download.sh
$ head -100 dev_v1.1.json |./text-embeddings/bin/convert-to-vespa.py  2> /dev/null
</pre>

After the above we have two new files in the working directory: _queries.txt_ and _vespa_feed.json_. The _queries.txt_ file contains
queries which have at least one relevant passage and in our head 100 sample that equals 97 queries and 796 passages (documents).

**Feed dumped data file** 
<pre>
$ java -jar $VESPA_HOME/lib/jars/vespa-http-client-jar-with-dependencies.jar --file vespa_feed.json --endpoint http://localhost:8080 
</pre>
**Run query evaluation of 5 different rank profiles**
The evaluation script runs all queries read from stdin and for each query execute n rank-profiles and compute the MRR@10 metric
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


