<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - document processing

Data written to Vespa pass through document processing,
where [indexing](https://docs.vespa.ai/en/indexing.html) is one example.
Applications can add custom processing, normally done before indexing.
This is done by adding a [Document Processor](https://docs.vespa.ai/en/document-processing.html).
Such processing is synchronous, and this is problematic for processing
that requires other resources with high latency -
this can saturate the threadpool.

This application demonstrates how to use _Progress.LATER_
and the asynchronous [Document API](https://docs.vespa.ai/en/document-api-guide.html). Summary:
- Document Processors: modify / enrich data in the feed pipeline
- Multiple Schemas: store different kinds of data, like different database tables
- Enrich data from multiple sources: here, look up data in one schema and add to another
- Document API: write asynchronous code to fetch data

Flow:
1. Feed album document with the _music_ schema
1. Look up in the _lyrics_ schema if album with given ID has lyrics stored
1. Store album with lyrics in the _music_ schema

![image](https://cloud.vespa.ai/assets/async-docproc.svg)

Refer to [getting-started-docproc](https://cloud.vespa.ai/en/getting-started-docproc) to try this application.
