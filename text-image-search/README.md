<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - Text-Image Search

This sample is an example of a text-to-image search application. Taking a textual query, such as "two
people bicycling", it will return images containing two people on bikes. This
application is built using [CLIP (Contrastive Language-Image
Pre-Training)](https://github.com/openai/CLIP) which enables "zero-shot prediction".
This means that the system can return sensible results for images it hasn't
seen during training, allowing it to process and index any image. In this
use case, we use the [Flickr8k](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names)
dataset, which was not used during training of the CLIP model.

This sample application can be used in two different ways. The first is by using a 
[Python-based search app](https://github.com/vespa-engine/sample-apps/tree/master/text-image-search/src/python/README.md),
which is suitable for exploration and analysis. The other is a 
[stand-alone Vespa application](https://github.com/vespa-engine/sample-apps/blob/master/text-image-search/README.md),
which is more suitable for production.

![Text-Image Search with Vespa](resources/demo.gif)

## Compare pre-trained CLIP models for text-image retrieval

This figure below shows the Reciprocal Rank @ 100 for each of the six 
available pre-trained CLIP models. Check [this notebook]((https://github.com/vespa-engine/sample-apps/blob/master/text-image-search/src/python/compare-pre-trained-clip-for-text-image-search.ipynb)) to see how 
to do the end-to-end analysis using [the Vespa python API](https://pyvespa.readthedocs.io/en/latest/index.html).

![alt text](resources/clip-evaluation-boxplot.png)

## Vespa CLI

**Validate environment, should be minimum 6G:**

<pre>
$ docker info | grep "Total Memory"
</pre>

**Check-out:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/text-image-search 
</pre>

**Set up transformer model:**
<pre data-test="exec">
$ pip3 -r src/python/requirements.txt
$ python3 src/python/clip_export.py
</pre>

**Compile and run:**

<pre data-test="exec">
$ mvn clean package
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 vespaengine/vespa
</pre>

**Wait for the configserver to start:**

<pre data-test="exec" data-test-wait-for="is ready">
$ vespa status deploy --wait 300 --color never
</pre>

**Deploy the application and wait for it to start:**

<pre data-test="exec" data-test-wait-for="is ready">
$ vespa deploy --wait 300 --color never
</pre>

**Download and extract image data:**

<pre data-test="exec">
$ ./src/sh/download_flickr8k.sh
$ export IMG_DIR=data/Flicker8k_Dataset/
</pre>

**Feed data:**

<pre data-test="exec">
$ python3 src/python/clip_feed.py
</pre>

**Search:**

<pre data-test="exec">
$ http://localhost:8080/search/?input=two+people
</pre>

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
