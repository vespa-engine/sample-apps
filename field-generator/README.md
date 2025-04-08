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

<p data-test="run-macro init-deploy field-generator">
Requires at least Vespa 8.507.34
</p>

## To try this application

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started)
through the <code>vespa deploy</code> step, cloning `field-generator` instead of `album-recommendation`.

Feed 10 documents (this includes generating fields values with LLM in Vespa):
<pre data-test="exec">
vespa feed data/feed_10.jsonl --connections 1 --verbose
</pre>

Query 10 documents to see generated values:
<pre data-test="exec" data-test-assert-contains="id:msmarco:passage::963">
vespa query 'yql=select * from passage where true' 'hits=10' 'ranking=enriched'
</pre>