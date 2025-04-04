<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Document enrichment with LLMs example

This sample application demonstrates how to use document enrichment with LLMs in Vespa.
See [Document enrichment with LLMs](https://docs.vespa.ai/en/llms-document-enrichment.html) documentation for detailed walkthrough of this app.
[Application schema](schemas/passage.sd) defines two fields with [generate indexing expression](reference/indexing-language-reference.html#generate).
Value for these fields are generated with an LLM during feeding.

Generators, LLMs and compute resources are configured in [services.xml](services.xml).
The default configuration uses a [local LLM](https://docs.vespa.ai/en/llms-local.html) and a GPU node in Vespa Cloud.
Comments in [services.xml](services.xml) contain instructions for reconfiguring this application including:
1. How to use an external LLM (OpenAI API) instead of a local LLM
2. Run the application locally instead of Vespa Cloud.
3. Use CPU nodes instead of GPU

## How to run this app

This application contains [Makefile](Makefile) with [vespa cli](https://docs.vespa.ai/en/vespa-cli.html)
commands to configure, deploy, feed and query the app.
By default, it is configured for Vespa Cloud.

Steps to run this app:

1. In the [Makefile](Makefile) change `tenant` to your tenant name.
2. Configure the environment: `make config`
3. Authenticate: `make auth`
4. Deploy the app: `make deploy`
5. Feed the app: `make feed-10`
6. Query the app: `make query-100`
7. Delete the app: `make destroy`
