
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa Code And Operational Examples

### Vespa grouping and facets for organizing results
The [part-purchases-demo](part-purchases-demo) app demonstrates Vespa grouping and faceting for query time result analytics. See [Vespa grouping documentation](https://docs.vespa.ai/en/grouping.html).

### Vespa predicate fields
The [predicate-fields](predicate-fields) demonstrates how to use Vespa's [predicate field](https://docs.vespa.ai/en/predicate-fields.html) type to implement indexing of document side boolean expressions.
Boolean document side constraints allows the document to specify which type of queries it can be retrieved for. Predicate fields allow expressing logic like _"this document should only be visible in search for readers in age range 20 to 30"_ or _"this product should only be visible in search during campaign hours"_.

### Operations
See [operations](operations) for sample applications for multinode clusters,
deployed in various infrastructure like Kubernetes. Also find examples for security and monitoring.

### Vespa custom linguistics Integration
The [vespa-chinese-linguistics](vespa-chinese-linguistics) app
demonstrates integrating custom linguistic processing,
in this case a Chinese tokenizer [Jieba](https://github.com/fxsjy/jieba).

### Vespa custom HTTP api using request handlers and processors
The [http-api-using-request-handlers-and-processors](http-api-using-request-handlers-and-processors)
demonstrates how to build custom HTTP apis, building REST interfaces with custom handlers and renderers.
See also [Custom HTTP Api tutorial](https://docs.vespa.ai/en/jdisc/http-api-tutorial.html).

### Vespa container plugins with multiple OSGI bundles
[multiple-bundles](multiple-bundles) is a technical sample application demonstrating how to use multiple OSGI bundles for custom plugins (searchers, handlers, renderers).

### Distributed joins 
[Joins](joins) shows possibilities for doing distributed query time joins.
This is for use cases where [parent-child](https://docs.vespa.ai/en/parent-child.html) is not sufficient. 

### [Document processing]
[Document-processing](document-processing)builds on album-recommendation to show
some of the possibilities for doing custom document processing in Java.

### Generic request processing
[generic-request-processing](generic-request-processing)
Generic [request-response](https://docs.vespa.ai/en/jdisc/processing.html) processing sample application.
<!-- ToDo: FIXME -->


### Lucene Linguistics
The [lucene-linguistics](lucene-linguistics) contains two sample application packages:
1. A bare minimal app.
2. Shows advanced configuration of the Lucene based `Linguistics` implementation.

----

Note: Applications with _pom.xml_ are Java/Maven projects and must be built before being deployed.
Refer to the [Developer Guide](https://docs.vespa.ai/en/developer-guide.html) for more information.

[Contribute](https://github.com/vespa-engine/vespa/blob/master/CONTRIBUTING.md) to the Vespa sample applications.

----

[![Vespa Sampleapps Search Feed](https://github.com/vespa-engine/sample-apps/actions/workflows/feed.yml/badge.svg)](https://github.com/vespa-engine/sample-apps/actions/workflows/feed.yml)

[![sample-apps link checker](https://api.screwdriver.cd/v4/pipelines/7038/link-checker-sample-apps/badge)](https://cd.screwdriver.cd/pipelines/7038/)

[![sample-apps build](https://api.screwdriver.cd/v4/pipelines/7038/build-apps/badge)](https://cd.screwdriver.cd/pipelines/7038/)

[![sample-apps verify-guides](https://api.screwdriver.cd/v4/pipelines/7038/verify-guides/badge)](https://cd.screwdriver.cd/pipelines/7038/)
