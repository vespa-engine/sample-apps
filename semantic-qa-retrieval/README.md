<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample application - Semantic Retrieval for Question-Answer Applications 

See [Build sentence/paragraph level QA application from python with Vespa](https://blog.vespa.ai/build-qa-app-from-python-with-vespa/). 

See also the more comprehensive [question answering sample app](../dense-passage-retrieval-with-ann/).

The sample application uses Vespa's
[Approximate Nearest Neighbor Search](https://docs.vespa.ai/en/approximate-nn-hnsw.html) support.

## Evaluation results for 87,599 questions

As reported in  [ReQA: An Evaluation for End-to-End Answer Retrieval Models](https://arxiv.org/abs/1907.04780)
versus the Vespa implementation for sentence level retrieval and paragraph level retrieval is given in the tables below:

**Sentence Level Retrieval**

|Model   | MRR  | R@1  | R@5  | R@10  |
|---|---|---|---|---|
|USE_QA for sentence answer retrieval | 0.539  | 0.439  | 0.656  | 0.727   |
|USE_QA on Vespa using tensors        | 0.538  | 0.438  | 0.656  | 0.726   |

**Paragraph Level Retrieval**

|Model   | MRR  | R@1  | R@5  | R@10  |
|---|---|---|---|---|
|USE_QA for paragraph answer retrieval | 0.634 | 0.533 | 0.757 | 0.823   |
|USE_QA on Vespa using tensors and Vespa grouping       | 0.633 | 0.532| 0.756  | 0.822|

On average the sentence embedding model described in the paper and realized on Vespa
has the sentence with the correct answer at the top 1 position in 44% of the questions for sentence level retrieval
over a collection of 91,729 sentences and 53% when doing paragraph retrieval over a collection of 18,896 paragraphs.

Some sample questions from the SQuAD v1.1 dataset is show below:

* Which NFL team represented the AFC at Super Bowl 50?
* What color was used to emphasize the 50th anniversary of the Super Bowl?
* What virus did Walter Reed discover?

One can explore the questions and the labeled answers [here](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/)

## Running this sample application 

**Requirements for running this sample application:**

* [Docker](https://www.docker.com/) installed and running  
* git client to checkout the sample application repository
* Operating system: macOS or Linux, Architecture: x86_64
* Minimum 6 GB memory dedicated to Docker (the default is 2 GB on Macs).
  Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
  for details and troubleshooting.
 
See also [Vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html).
This setup is slightly different then the official quick start guide
as we build a custom docker image  with the tensorflow dependencies.

**Validate environment, should be minimum 6G:**

<pre>
$ docker info | grep "Total Memory"
</pre>


**Checkout the sample-apps repository**

This step requires that you have a working git client:
<pre>
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git; cd sample-apps/semantic-qa-retrieval
</pre>


**Build a docker image (See [Dockerfile](Dockerfile) for details)**

The image builds on the [vespaengine/vespa docker image (latest)](https://hub.docker.com/r/vespaengine/vespa/tags)
and installs python3 and the python dependencies to run tensorflow.
This step takes a few minutes.
<pre>
$ docker build . --tag vespa_semantic_qa_retrieval:1.0
</pre>


**Run the docker container built in the previous step and enter the running docker container**

<pre>
$ docker run --detach --name vespa_qa --hostname vespa-container vespa_semantic_qa_retrieval:1.0
$ docker exec -it vespa_qa bash 
</pre>


**Deploy the document schema and configuration - this will start Vespa services**

<pre>
$ vespa-deploy prepare qa/src/main/application/ && vespa-deploy activate
</pre>


**Download the SQuaAD train v1.1 dataset and convert format to Vespa (Sample of 269 questions)**

The download script will extract a sample set
(As processing the whole dataset using the Sentence Encoder for QA takes time).

<pre>
$ ./qa/bin/download.sh
$ ./qa/bin/convert-to-vespa-squad.py sample_squad.json 2> /dev/null
</pre>

After the above we have two new files in the working directory: 
_squad_queries.txt_ and _squad_vespa_feed.json_.
The _queries.txt_ file contains question.
The sample question set generates 351 sentence documents and 55 context documents (The paragraphs).


**Feed Vespa json** 

We feed the documents using the [vespa-feed-client](https://docs.vespa.ai/en/vespa-feed-client.html):
<pre>
$ $VESPA_HOME/bin/vespa-feed-client --file squad_vespa_feed.json --endpoint http://localhost:8080
</pre>


**Run evaluation**

The evaluation script runs all questions produced by the conversion script
and for each question it executes different recall and ranking strategies and finally it computes the
[mean reciprocal rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) _MRR@100_
and the Recall@1,Recall@5 and Recall@10 metrics.

The evaluations script uses the [Vespa query api](https://docs.vespa.ai/en/query-api.html)

Running the _evaluation.py_ script:

<pre>
$ cat squad_queries.txt | ./qa/bin/evaluation.py
</pre>

Which should produce output like this:

<pre>
Start query evaluation for 269 queries
Sentence retrieval metrics:
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   MRR@100  0.5799
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@1 0.4498
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@5 0.7398
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@10 0.8290
Paragraph retrieval metrics:
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   MRR@100  0.7030
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@1 0.5725
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@5 0.8625
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@10 0.9405
</pre>


**Reproducing the paper metrics**

To reproduce the paper one need to convert the entire dataset and do evaluation over all questions:

<pre>
$ ./qa/bin/convert-to-vespa-squad.py SQuAD_train_v1.1.json 2> /dev/null
$ $VESPA_HOME/bin/vespa-feed-client --file squad_vespa_feed.json --endpoint http://localhost:8080
$ cat squad_queries.txt |./qa/bin/evaluation.py 2> /dev/null
</pre>

Which should produce output like this: 
<pre>
Start query evaluation for 87599 queries
Sentence retrieval metrics:
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   MRR@100  0.5376
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@1 0.4380
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@5 0.6551
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@10 0.7262
Paragraph retrieval metrics:
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   MRR@100  0.6330
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@1 0.5322
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@5 0.7555
Profile 'sentence-semantic-similarity', doc='sentence', dataset='squad',   R@10 0.8218
</pre>
