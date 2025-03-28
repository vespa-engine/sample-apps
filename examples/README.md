
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa Code And Operational Examples

### Vespa grouping and facets for organizing results
[![logo](/assets/vespa-logomark-tiny.png) Grouping Results](part-purchases-demo)
demonstrates Vespa grouping and faceting for query time result analytics.
Read more in [Vespa grouping](https://docs.vespa.ai/en/grouping.html).


### Vespa Predicate Fields
[![logo](/assets/vespa-logomark-tiny.png) predicate-fields](predicate-fields)
uses Vespa's [predicate field](https://docs.vespa.ai/en/predicate-fields.html) type to implement indexing of document side boolean expressions.
Boolean document side constraints allows the document to specify which type of queries it can be retrieved for.
Predicate fields allow expressing logic like _"this document should only be visible in search for readers in age range 20 to 30"_ or
_"this product should only be visible in search during campaign hours"_.


### Vespa custom linguistics Integration
The [![logo](/assets/vespa-logomark-tiny.png) vespa-chinese-linguistics](vespa-chinese-linguistics) app
demonstrates integrating custom linguistic processing,
in this case a Chinese tokenizer [Jieba](https://github.com/fxsjy/jieba).


### Vespa custom HTTP api using request handlers and processors
[![logo](/assets/vespa-logomark-tiny.png) http-api-using-request-handlers-and-processors](http-api-using-request-handlers-and-processors)
demonstrates how to build custom HTTP apis, building REST interfaces with custom handlers and renderers.
See also [Custom HTTP Api tutorial](https://docs.vespa.ai/en/jdisc/http-api-tutorial.html).


### Vespa container plugins with multiple OSGI bundles
[![logo](/assets/vespa-logomark-tiny.png) multiple-bundles](multiple-bundles) is a technical sample application
demonstrating how to use multiple OSGI bundles for custom plugins (searchers, handlers, renderers).


### Distributed joins 
[![logo](/assets/vespa-logomark-tiny.png) Joins](joins) shows possibilities for doing distributed query time joins.
This is for use cases where [parent-child](https://docs.vespa.ai/en/parent-child.html) is not sufficient. 


### Document processing
[![logo](/assets/vespa-logomark-tiny.png) Document-processing](document-processing)builds on
[album-recommendation](/album-recommendation) to show
some of the possibilities for doing custom document processing in Java.


### Generic request processing
[![logo](/assets/vespa-logomark-tiny.png) generic-request-processing](generic-request-processing)
Generic [request-response](https://docs.vespa.ai/en/jdisc/processing.html) processing sample application.
<!-- ToDo: FIXME -->


### Lucene Linguistics
[![logo](/assets/vespa-logomark-tiny.png) lucene-linguistics](lucene-linguistics) contains two sample application packages:
1. A bare minimal app.
2. Shows advanced configuration of the Lucene based `Linguistics` implementation.


### Lambda functions in AWS and Google Cloud
[![logo](/assets/vespa-logomark-tiny.png) aws/lambda](aws/lambda) and
[![logo](/assets/vespa-logomark-tiny.png) google-cloud/cloud-functions](google-cloud/cloud-functions)
have examples of (lambda) functions for accessing data and logs with the cloud providers.


### Automatic data generation for training embedders using LLMs
[![logo](/assets/vespa-logomark-tiny.png) embedder-auto-training-evaluation](embedder-auto-training-evaluation) does
automatic data generation using the ChatGPT API.
This in order to train an embedder to perform better for information retrieval on specific datasets
without labor-intensive and expensive manual training data annotation.

Machine learned embedder models enable efficient similarity computations,
but training these models requires large amounts of (often manually) annotated data.
The aim of this app is to investigate whether Large Language Models (LLMs),
such as GPT-3.5-turbo, can be employed to generate synthetic data for training embedder,
without extensive manual intervention.

The repository contains scripts and notebooks to:
* Prepare datasets
* Generate training data for datasets using an LLM
* Train an embedder
* Evaluate performance

Read more in the [blog post](https://blog.vespa.ai/summer-internship-2023/#automatic-embedder-training-with-an-llm).


### Embedding service (WORK IN PROGRESS)
[![logo](/assets/vespa-logomark-tiny.png) embedding-service](embedding-service)
demonstrates how a Java handler component can be used to process HTTP requests.
In this application, a handler is used to implement an embedding service,
which takes a string as an input and returns a vector embedding of that string.


### FastHTML Vespa frontend
[![logo](/assets/vespa-logomark-tiny.png) FastHTML Vespa frontend](fasthtml-demo)
is a simple frontend for the Vespa search engine.
It is built using [FastHTML](https://www.fastht.ml/) and written in pure Python. Features:
* Simple search interface, with links to search results.
* Accordion with full JSON-response from Vespa.
* SQLite DB for storing queries.
* Admin authentication for viewing and downloading queries.
* Deployment options - Docker + Huggingface spaces.


### ONNX Model export and deployment example
Use [![logo](/assets/vespa-logomark-tiny.png) model-deployment](model-deployment) to generate a model in ONNX format in the models directory,
by running the ONNXModelExport notebook.
<!-- ToDo: improve this -->


### Model exporting
[![logo](/assets/vespa-logomark-tiny.png) Model exporting](model-exporting)
demonstrates how to export a Huggingface sentence-transformer model to ONNX format.


### Reranker sample application
[![logo](/assets/vespa-logomark-tiny.png) reranker](reranker) is a stateless application which re-ranks results obtained from another Vespa application.
While this does not result in good performance and is not recommended for production,
it is useful when you want to quickly do ranking experiments without rewriting application data.


### Categorize using an LLM
[![logo](/assets/vespa-logomark-tiny.png) In-Context Learning](in-context-learning) This is a set of scripts/installs to back up the presentation using In-Context Learning at:
* [MLCon](https://mlconference.ai/machine-learning-advanced-development/adaptive-incontext-learning/)
* [data science connect COLLIDE](https://datasciconnect.com/events/collide/agenda/)

### Agentic Chatbot using Vespa

[![logo](/assets/vespa-logomark-tiny.png) agentic-streamlit-chatbot](agentic-streamlit-chatbot/simple_app) This simple Streamlit application demonstrates how to use [LangGraph](https://www.langchain.com/langgraph) agentic framework to develop an E-commerce chatbot using Vespa as a retrieval tool.

[![logo](/assets/vespa-logomark-tiny.png) agentic-streamlit-chatbot](agentic-streamlit-chatbot/advanced_app) This Streamlit application shows a more advanced example on how to use [LangGraph](https://www.langchain.com/langgraph) agentic framework to develop an E-commerce chatbot enabling a conversational search with human in the loop feedback with yql query generation using Vespa query builder.

For any questions, please register at the Vespa Slack and discuss in the general channel.

----

### Operations
See [operations](operations) for sample applications for multinode clusters,
deployed in various infrastructure like Kubernetes.
Also find examples for CI/CD, security and monitoring.


Note: Applications with _pom.xml_ are Java/Maven projects and must be built before being deployed.
Refer to the [Developer Guide](https://docs.vespa.ai/en/developer-guide.html) for more information.

[Contribute](https://github.com/vespa-engine/vespa/blob/master/CONTRIBUTING.md) to the Vespa sample applications.
