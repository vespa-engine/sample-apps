<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample applications - Text/Image search

Build a text to image search engine from scratch with Vespa.

![Text-Image Search with Vespa](resources/demo.gif)

## Compare pre-trained CLIP models for text-image retrieval

Figure below shows the Reciprocal Rank @ 100 for each of the six 
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
