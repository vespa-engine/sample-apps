<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample applications - CLIP

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
$ mkdir -p src/main/application/models
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
$ vespa status deploy --wait 300
</pre>

**Deploy the application and wait for it to start:**

<pre data-test="exec" data-test-wait-for="is ready">
$ vespa deploy --wait 300
</pre>

**Feed data:**

TBD

**Search:**

TBD

**Shutdown and remove the container:**

<pre data-test="after">
$ docker rm -f vespa
</pre>
