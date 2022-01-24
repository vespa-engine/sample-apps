<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - album recommendation, with Java components

Vespa applications can contain Java components which are run inside Vespa to implement the
functionality required by the application.
This sample application is the same as album-recommendation,
but with some Java components, and the maven setup to build them added to it.

The Java components added here are of the most common type, 
[searchers](https://docs.vespa.ai/en/searcher-development.html),
which can modify the query and result, issue multiple queries for ech request etc.
There are also many other component types,
such as [document processors](https://docs.vespa.ai/en/document-processing.html), 
which can modify document data as it is written to Vespa,
and [handlers](https://docs.vespa.ai/en/jdisc/developing-request-handlers.html),
which can be used to let Vespa expose custom service APIs.

Follow to [Quick start, with Java](https://docs.vespa.ai/en/vespa-quick-start-java.html) 
to build and deploy this sample application.
