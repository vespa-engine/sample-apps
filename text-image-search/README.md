<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample application - Text-Image Search

This sample is an example of a text-to-image search application. Taking a textual query, such as "two
people bicycling", it will return images containing two people on bikes. This
application is built using [CLIP (Contrastive Language-Image
Pre-Training)](https://github.com/openai/CLIP) which enables "zero-shot prediction".
This means that the system can return sensible results for images it hasn't
seen during training, allowing it to process and index any image. In this
use case, we use the [Flickr8k](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names)
dataset, which was not explicitly used during training of the CLIP model.

This sample application can be used in two different ways.
The first is by using a [Python-based search app](src/python/README.md), which is suitable for exploration and analysis.
The other is a stand-alone Vespa application, which is more suitable for production (below).

The Python sample app includes a streamlit user interface:

![Text-Image Search with Vespa](resources/demo.gif)


## Vespa CLI

The following instructions sets up the stand-alone Vespa application using the
Vespa CLI, which takes a textual description and returns the file names of the
images that best match the description. The main difference between this app
and the Python app, is that the transformation from text to a vector
representation has been moved from Python and into Vespa. This includes both
tokenization and transformer model evaluation.

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
$ pip3 install -r src/python/requirements.txt
$ python3 src/python/clip_export.py
</pre>

This extracts the text transformer model from CLIP and puts it into the
`models` directory of the application.

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

The full Flickr8k dataset is around 1.1Gb.

**Feed data:**

<pre data-test="exec">
$ python3 src/python/clip_feed.py
</pre>

This uses pyvespa to feed the image data to the running instance.

**Search:**

<pre data-test="exec">
$ http://localhost:8080/search/?input=a+child+playing+football
</pre>

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
