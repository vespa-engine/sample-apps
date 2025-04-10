<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Document enrichment with LLMs example

This sample app demonstrates how to use document enrichment with LLMs in Vespa.
[App's schema](schemas/passage.sd) defines two fields using [generate indexing expression](https://docs.vespa.ai/en/reference/indexing-language-reference.html#generate).
Values for these fields are generated with an LLM during feeding.

The LLM, generator components and compute resources are configured in [services.xml](services.xml).
The default configuration uses a [local LLM](https://docs.vespa.ai/en/llms-local.html) and a GPU node in Vespa Cloud.
See comments in [services.xml](services.xml) for instructions to reconfigure this app for the following scenarios:

1. Replace local LLM with external LLM (OpenAI API)
2. Run the app locally instead of Vespa Cloud
3. Use CPU nodes instead of GPU

See [document enrichment with LLMs](https://docs.vespa.ai/en/llms-document-enrichment.html) documentation for detailed walkthrough.

## To try this application

The steps blow deploy and test this app in a docker container locally.
It is also possible to deploy it in [Vespa Cloud](https://docs.vespa.ai/en/cloud/getting-started).

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

Pull and start the vespa docker container image:
<pre data-test="exec">
$ docker pull vespaengine/vespa
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
</pre>

Download this sample application:
<pre data-test="exec">
$ vespa clone field-generator myapp && cd myapp
</pre>

Verify that configuration service (deploy api) is ready:
<pre data-test="exec">
$ vespa status deploy --wait 300
</pre>

Deploy the application (it takes extra time to download the model):
<pre data-test="exec" data-test-assert-contains="Success">
$ vespa deploy --wait 900
</pre>

Wait for the application endpoint to become available:
<pre data-test="exec">
$ vespa status --wait 300
</pre>

Feed 10 documents (this includes generating fields values with LLM in Vespa):
<pre data-test="exec">
vespa feed data/feed_10.jsonl --connections 1 --verbose
</pre>

Query 10 documents to see generated values:
<pre data-test="exec" data-test-assert-contains="id:msmarco:passage::963">
vespa query 'yql=select * from passage where true' 'hits=10' 'ranking=enriched'
</pre>

Clean up after test:
<pre data-test="after">
$ docker rm -f vespa
</pre>
