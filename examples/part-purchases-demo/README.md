<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Grouping Results

Vespa grouping is a powerful feature that allows you to group search results based on field values.
This is useful when you want to present search results in a structured way,
such as grouping search results by category or price range.
Grouping can be used to create faceted search results and to aggregate data.

Read more in the [Grouping](https://docs.vespa.ai/en/grouping.html) guide.

----

**Requirements:**

* [Docker](https://www.docker.com/) Desktop installed and running. 4GB available memory for Docker is recommended.
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement)
* Architecture: x86_64 or arm64
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or download
  a vespa cli release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).


**Validate environment, should be minimum 4G:**
<pre>
$ docker info | grep "Total Memory"
or
$ podman info | grep "memTotal"
</pre>

Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
for details and troubleshooting:


**Check-out, start Docker container:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/examples/part-purchases-demo
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
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
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>


**Deploy the application:**
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 300
</pre>


**Feed to Vespa**
<pre data-test="exec">
$ vespa feed ext/feed.jsonl
</pre>


**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>

----

**Feed the data with Logstash from the CSV file**

You can also feed the data with Logstash from the CSV file directly (no need to run `parts.py`). It may be useful if you want to feed your own data.

You'll need to [install Logstash](https://www.elastic.co/downloads/logstash), then:

1. Install the [Logstash Output Plugin for Vespa](https://github.com/vespa-engine/vespa/tree/master/integration/logstash-plugins/logstash-output-vespa) via:

```
bin/logstash-plugin install logstash-output-vespa_feed
```

2. Adapt [this logstash.conf](https://github.com/vespaai/university/blob/main/101/ch5/logstash.conf) to point to the absolute path of [purchase.csv](ext/purchase.csv).

3. Run Logstash with the modified `logstash.conf`:

```
bin/logstash -f $PATH_TO_LOGSTASH_CONF/logstash.conf
```

For more examples of using Logstash with Vespa, check out [this tutorial blog post](https://blog.vespa.ai/logstash-vespa-tutorials/).
