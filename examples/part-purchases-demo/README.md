<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - Part Purchases Demo

A sample Vespa application to assist with learning how to group according to the
[Grouping Guide](https://docs.vespa.ai/en/grouping.html).

**Requirements:**

* [Docker](https://www.docker.com/) Desktop installed and running. 4GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).


**Validate environment, should be minimum 4G:**

<pre>
$ docker info | grep "Total Memory"
</pre>

Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
for details and troubleshooting:


**Check-out, start Docker container:**

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/examples/part-purchases-demo
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
</pre>


**Install the Vespa CLI, configure:**
<pre>
$ brew install vespa-cli
</pre>
<pre data-test="exec">
$ vespa config set target local
</pre>


**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="Deploy API at http://127.0.0.1:19071 is ready">
$ vespa status deploy --wait 300
</pre>


**Deploy the application:**
<pre data-test="exec" data-test-assert-contains="Success: Deployed">
$ vespa deploy --wait 300
</pre>


**Generate sample from csv**
<pre data-test="exec">
$ python3 ext/parts.py -f ext/purchase.csv > ext/purchase.json
</pre>


**Feed data into application:**
<pre data-test="exec" data-test-wait-for='"feeder.ok.count": 20'>
$ vespa feed ext/purchase.json
</pre>


**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
