<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Billion-Scale Image Search

This sample application combines two sample applications to implement
cost-efficient large scale image search over multimodal AI powered vector representations;
[text-image-search](https://github.com/vespa-engine/sample-apps/tree/master/text-image-search) and
[billion-scale-vector-search](https://github.com/vespa-engine/sample-apps/tree/master/billion-scale-vector-search).

## The Vector Dataset
This sample app use the [LAION-5B](https://laion.ai/blog/laion-5b/) dataset,
 the biggest open accessible image-text dataset in the world.

> Large image-text models like ALIGN, BASIC, Turing Bletchly, FLORENCE & GLIDE have
> shown better and better performance compared to previous flagship models like CLIP and DALL-E.
> Most of them had been trained on billions of image-text pairs and unfortunately, no datasets of this size had been openly available until now.
> To address this problem we present LAION 5B, a large-scale dataset for research purposes
> consisting of 5,85B CLIP-filtered image-text pairs. 2,3B contain English language,
> 2,2B samples from 100+ other languages and 1B samples have texts that do not allow a certain language assignment (e.g. names ).

The LAION-5B dataset was used to train the popular text-to-image
generative [StableDiffusion](https://stability.ai/blog/stable-diffusion-public-release) model.

Note the following about the LAION 5B dataset

> Be aware that this large-scale dataset is un-curated.
> Keep in mind that the un-curated nature of the dataset means that collected
> links may lead to strongly discomforting and disturbing content for a human viewer.

The released dataset does not contain image data itself,
but <a href="https://openai.com/research/clip" data-proofer-ignore>CLIP</a> encoded vector representations of the images,
and metadata like `url` and `caption`.

## Use cases

The app can be used to implement several use cases over the LAION dataset, or adopted to your large-scale vector dataset:

- Search with a free text prompt over the `caption` or `url` fields in the LAION dataset using Vespa's standard text-matching functionality.
- CLIP retrieval, using vector search, given a text prompt, search the image vector representations (CLIP ViT-L/14), for example for 'french cat'.
- Given an image vector representation, search for similar images in the dataset. This can for example
be used to take the output image of StableDiffusion to find similar images in the training dataset.

All this combined using [Vespa's query language](https://docs.vespa.ai/en/query-language.html),
 and also in combination with filters.

## Vespa Primitives Demonstrated

The sample application demonstrates many Vespa primitives:

- Importing an [ONNX](https://onnx.ai/)-exported version of [CLIP ViT-L/14](https://github.com/openai/CLIP)
for [accelerated inference](https://blog.vespa.ai/stateful-model-serving-how-we-accelerate-inference-using-onnx-runtime/)
in [Vespa stateless](https://docs.vespa.ai/en/overview.html) containers.
The exported CLIP model encodes a free-text prompt to a joint image-text embedding space with 768 dimensions.
- [HNSW](https://docs.vespa.ai/en/approximate-nn-hnsw.html) indexing of vector centroids drawn
from the dataset, and combination with classic Inverted File as described in
[Billion-scale vector search using hybrid HNSW-IF](https://blog.vespa.ai/vespa-hybrid-billion-scale-vector-search/).
- Decoupling of vector storage and vector similarity computations. The stateless layer performs vector
similarity computation over the full precision vectors.
By using Vespa's support for accelerated inference with [onnxruntime](https://onnxruntime.ai/),
moving the majority of the vector compute to the stateless layer
allows for faster auto-scaling with daily query volume changes.
The full precision vectors are stored in Vespa's summary log store, using lossless compression (zstd).
- Dimension reduction with PCA - The centroid vectors are compressed from 768 dimensions to 128 dimensions. This allows indexing 6x more
centroids on the same instance type due to the reduced memory footprint. With Vespa's support for distributed search, coupled with powerful
high memory instances, this allows Vespa to scale cost efficiently to trillion-sized vector datasets.
- The trained PCA matrix and matrix multiplication which projects the 768-dim vectors to 128-dimensions is
evaluated in Vespa using accelerated inference, both at indexing time and at query time. The PCA weights are represented also using ONNX.
- Phased ranking.
The image embedding vectors are also projected to 128 dimensions, stored using
memory mapped [paged attribute tensors](https://docs.vespa.ai/en/attributes.html#paged-attributes).
Full precision vectors are on stored on disk in Vespa summary store.
The first-phase coarse search ranks vectors in the reduced vector space, per node, and results are merged from all nodes before
the final ranking phase in the stateless layer.
The final ranking phase is implemented in the stateless container layer using [accelerated inference](https://blog.vespa.ai/stateful-model-serving-how-we-accelerate-inference-using-onnx-runtime/).
- Combining approximate nearest neighbor search with [filters](https://blog.vespa.ai/constrained-approximate-nearest-neighbor-search/), filtering
can be on url, caption, image height, width, safety probability, NSFW label, and more.
- Hybrid ranking, both textual sparse matching features and the CLIP similarity, can be used when ranking images.
- Reduced tensor cell precision. The original LAION-5B dataset uses `float16`. The app uses Vespa's support for `bfloat16` tensors,
  saving 50% of storage compared to full `float` representation.
- Caching, both reduced vectors (128) cached by the OS buffer cache, and full version 768 dims are cached using Vespa summary cache.
- Query-time vector de-duping and diversification of the search engine result page using document to document similarity instead of query to document similarity. Also
accelerated by stateless model inference.
- Scale, from a single node deployment to multi-node deployment using managed [Vespa Cloud](https://cloud.vespa.ai/),
or self-hosted on-premise.


## Stateless Components
The app contains several [container components](https://docs.vespa.ai/en/jdisc/container-components.html):

- [RankingSearcher](src/main/java/ai/vespa/examples/searcher/RankingSearcher.java) implements the last stage ranking using
full-precision vectors using an ONNX model for accelerated inference.
- [DedupingSearcher](src/main/java/ai/vespa/examples/searcher/DeDupingSearcher.java) implements run-time de-duping after Ranking, using
document to document similarity matrix, using an ONNX model for accelerated inference.
- [DimensionReducer](src/main/java/ai/vespa/examples/DimensionReducer.java) PCA dimension reducing vectors from 768-dims to 128-dims.
- [AssignCentroidsDocProc](src/main/java/ai/vespa/examples/docproc/AssignCentroidsDocProc.java) searches the HNSW graph content cluster
during ingestion to find the nearest centroids of the incoming vector.
- [SPANNSearcher](src/main/java/ai/vespa/examples/searcher/SPANNSearcher.java)

## Deploying this app
These reproducing steps, demonstrates the app using a smaller subset of the LAION-5B vector dataset, suitable
for playing around with the app on a laptop.

**Requirements:**

* [Docker](https://www.docker.com/) Desktop installed and running. 6GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Alternatively, deploy using [Vespa Cloud](#deployment-note)
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* <a href="https://openjdk.org/projects/jdk/17/" data-proofer-ignore>Java 17</a> installed.
* Python3 and numpy to process the vector dataset
* [Apache Maven](https://maven.apache.org/install.html) - this sample app uses custom Java components and Maven is used
  to build the application.

Verify Docker Memory Limits:

<pre>
$ docker info | grep "Total Memory"
or
$ podman info | grep "memTotal"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

Use the [multi-node high availability](https://github.com/vespa-engine/sample-apps/tree/master/examples/operations/multinode-HA)
template for inspiration for multi-node, on-premise deployments.

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
</pre>

Verify that the configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Download this sample application:
<pre data-test="exec">
$ vespa clone billion-scale-image-search myapp && cd myapp
</pre>


## Download Vector + Metadata

These instructions use the first split file (0000) of a total of 2314 files in the LAION2B-en split.
Download the vector data file:

<pre data-test="exec">
$ curl --http1.1 -L -o img_emb_0000.npy \
  https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/img_emb_0000.npy
</pre>

Download the metadata file:

<pre data-test="exec">
$ curl -L -o metadata_0000.parquet \
  https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/laion2B-en-metadata/metadata_0000.parquet
</pre>

Install python dependencies to process the files:

<pre data-test="exec">
$ python3 -m pip install pandas numpy requests mmh3 pyarrow
</pre>

Generate centroids, this process randomly selects vectors from the dataset to represent
centroids. Performing an incremental clustering can improve vector search recall and allow
indexing fewer centroids. For simplicity, this tutorial uses random sampling.

<pre data-test="exec">
$ python3 src/main/python/create-centroid-feed.py img_emb_0000.npy > centroids.jsonl
</pre>

Generate the image feed, this merges the embedding data with the metadata and creates a Vespa
jsonl feed file, with one json operation per line.

<pre data-test="exec">
$ python3 src/main/python/create-joined-feed.py metadata_0000.parquet img_emb_0000.npy > feed.jsonl
</pre>

To process the entire dataset, we recommend starting several processes, each operating on separate split files
as the processing implementation is single-threaded.


## Build and deploy Vespa app

`src/main/application/models` has three small ONNX models:

- `vespa_innerproduct_ranker.onnx` for vector similarity (inner dot product) between the query and the vectors
in the stateless container.
- `vespa_pairwise_similarity.onnx` for matrix multiplication between the top retrieved vectors.
- `pca_transformer.onnx` for dimension reduction, projecting the 768-dim vector space to a 128-dimensional space.

These `ONNX` model files are generated by specifying the compute operation using [pytorch](https://pytorch.org/) and using `torch`'s
ability to export the model to [ONNX](https://onnx.ai/) format:

- [ranker_export.py](src/main/python/ranker_export.py)
- [similarity_export.py](src/main/python/similarity_export.py)
- [pca_transformer_export.py](src/main/python/pca_transformer_export.py)

Build the sample app (make sure you have JDK 17, verify with `mvn -v`): - This step
also downloads a pre-exported ONNX model for mapping the prompt text to the CLIP vector embedding space.

<pre data-test="exec" data-test-expect="BUILD SUCCESS" data-test-timeout="300">
$ mvn clean package -U
</pre>

Deploy the application. This step deploys the application package built in the previous step:

<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>

#### Deployment note
It is possible to deploy this app to
[Vespa Cloud](https://cloud.vespa.ai/en/getting-started-java#deploy-sample-applications-java).
For Vespa cloud deployments to [perf env](https://cloud.vespa.ai/en/reference/zones.html)
replace the [src/main/application/services.xml](src/main/application/services.xml) with
[src/main/application/services-cloud.xml](src/main/application/services-cloud.xml) -
the cloud deployment uses dedicated clusters for `feed` and `query`.

Wait for the application endpoint to become available:

<pre data-test="exec">
$ vespa status --wait 300
</pre>

Run [Vespa System Tests](https://docs.vespa.ai/en/reference/testing.html),
which runs a set of basic tests to verify that the application is working as expected:
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa test src/test/application/tests/system-test/feed-and-search-test.json
</pre>

The _centroid_ vectors **must** be indexed first:

<pre data-test="exec">
$ vespa feed centroids.jsonl
$ vespa feed feed.jsonl
</pre>

Track number of documents while feeding:

<pre data-test="exec">
$ vespa query 'yql=select * from image where true' \
  hits=0 \
  ranking=unranked
</pre>


## Fetching data

Fetch a single document using [document api](https://docs.vespa.ai/en/reference/document-v1-api-reference.html):

<pre data-test="exec" data-test-assert-contains="vector">
$ vespa document get \
 id:laion:image::5775990047751962856
</pre>

The response contains all fields, including the full vector representation and the
reduced vector, plus all the metadata. Everything represented in the same
[schema](src/main/application/schemas/image.sd).


## Query the data
The following provides a few query examples,
`prompt` is a run-time query parameter which is used by the
[CLIPEmbeddingSearcher](src/main/java/ai/vespa/examples/searcher/CLIPEmbeddingSearcher.java)
which will encode the prompt text into a CLIP vector representation using the embedded CLIP model:

<pre data-test="exec" data-test-assert-contains="documentid">
$ vespa query \
 'yql=select documentid, caption, url, height, width from image where nsfw contains "unlikely"'\
 'hits=10' \
 'prompt=two dogs running on a sandy beach'
</pre>

Results are filtered by a constraint on the `nsfw` field. Note that even if the image is classified
as `unlikely` the image content might still be explicit as the NSFW classifier is not 100% accurate.

The returned images are ranked by CLIP similarity (The score is found in each hit's `relevance` field).

The following query adds another filter, restricting the search so that only images crawled from urls with `shutterstock.com`
is retrieved.

<pre data-test="exec" data-test-assert-contains="documentid">
$ vespa query \
 'yql=select documentid, caption, url, height, width from image where nsfw contains "unlikely" and url contains "shutterstock.com"'\
 'hits=10' \
 'prompt=two dogs running on a sandy beach'
</pre>

Another restricting the search further, adding a phrase constraint `caption contains phrase("sandy", "beach")`:

<pre data-test="exec" data-test-assert-contains="documentid">
$ vespa query \
 'yql=select documentid, caption, url, height, width from image where nsfw contains "unlikely" and url contains "shutterstock.com" and caption contains phrase("sandy", "beach")'\
 'hits=10' \
 'prompt=two dogs running on a sandy beach'
</pre>

Regular query, matching over the `default` fieldset, searching the `caption` and the `url` field, ranked by
the `text` ranking profile:

<pre data-test="exec" data-test-assert-contains="documentid">
$ vespa query \
 'yql=select documentid, caption, url from image where nsfw contains "unlikely" and userQuery()'\
 'hits=10' \
 'query=two dogs running on a sandy beach' \
 'ranking=text'
</pre>

The `text` [rank](https://docs.vespa.ai/en/ranking.html) profile uses
[nativeRank](https://docs.vespa.ai/en/nativerank.html), one of Vespa's many
text matching rank features.

## Non-native hyperparameters
There are several non-native query request
parameters that controls the vector search accuracy and performance tradeoffs. These
can be set with the request, e.g, `/search/&spann.clusters=12`.

- `spann.clusters`, default `64`, the number of centroids in the reduced vector space used to restrict the image search.
A higher number improves recall, but increases computational complexity and disk reads.
- `rank-count`,  default `1000`, the number of vectors that are fully re-ranked in the container using the full vector representation.
A higher number improves recall, but increases the computational complexity and network.
- `collapse.enable`, default `true`, controls de-duping of the top ranked results using image to image similarity.
- `collapse.similarity.max-hits`, default `1000`, the number of top-ranked hits to perform de-duping of. Must be less than `rank-count`.
- `collapse.similarity.threshold`, default `0.95`, how similar a given image to image must be before it is considered a duplicate.

## Areas of improvement
There are several areas that could be improved.

- CLIP model. The exported text transformer model uses fixed sequence length (77), this wastes computations and makes
the model a lot slower than it has to be for shorter sequence lengths. A dynamic sequence length would
make encoding short queries a lot faster than the current model.
It would also be interesting to use the text encoder as a teacher and train a smaller distilled model using a different architecture (for example based on smaller MiniLM models).
- CLIP query embedding caching. The CLIP model is fixed and only uses the text input. Caching the map from text to
embedding would save resources.

## Shutdown and remove the container:

<pre data-test="after">
$ docker rm -f vespa
</pre>
