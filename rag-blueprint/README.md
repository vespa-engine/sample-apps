<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# The RAG Blueprint

Vespa is the [platform of choice](https://blog.vespa.ai/perplexity-builds-ai-search-at-scale-on-vespa-ai/) for large scale RAG applications like Perplexity.
It gives you all the features you need but putting them all together can be a challenge.

This open source sample applications contains all the elements you need to create a RAG application that

* delivers state-of-the-art quality, and
* scales to any amount of data, query load, and complexity.

This README provides the steps to create and run your own application based on the blueprint.
Refer to the [RAG Blueprint tutorial](https://docs.vespa.ai/en/tutorials/rag-blueprint.html) for more in-depth explanations, or try out the [Python notebook](https://vespa-engine.github.io/pyvespa/examples/rag-blueprint-vespa-cloud.html).

## Create and run your own RAG application

Install the [Vespa-CLI](https://docs.vespa.ai/en/vespa-cli.html) client
(available for Linux, macOS and Windows, see the link if you don't have Homebrew).

<pre>
$ brew install vespa-cli
</pre>

Create your application from the blueprint:

<pre>
$ vespa clone rag-blueprint [your-application-name]
$ cd [your-application-name]
</pre>

If you don't have one,
[create a Vespa Cloud tenant for free](https://console.vespa-cloud.com/),
and configure Vespa CLI to deploy to it:

<pre>
$ vespa config set target cloud
$ vespa config set application [your-tenant-name].[your-application-name]
$ vespa auth cert app
</pre>

Deploy the application.

<pre>
$ vespa deploy app
</pre>

Feed some documents, this will also chunk and embed so it takes about 3 minutes:

<pre>
$ vespa feed dataset/docs.jsonl
</pre>

Now you can issue queries:

<pre>
$ vespa query 'query=yc b2b sales'
</pre>

> [!TIP]
> Add "-v" to see the HTTP request this becomes.

Congratulations! You have now created a RAG application that can scale to billions of documents and thousands
of queries per second, while delivering state-of-the-art quality.

## Explore more

What do you want to do next?

- To learn what this application can do, look at the files in your app/ dir.
- [Run your application locally using Docker](deploy-locally.md)
- [Using query profiles to define behavior for different use cases](query-profiles.md)
- [Evaluate and improve relevance of the data returned](relevance.md)
- [Do LLM generation inside the application](generation.md)
