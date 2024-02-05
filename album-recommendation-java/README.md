
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - album recommendation, with Java components

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started) to deploy this.

## Introduction

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


## Query tracing
See [MetalSearcher::search()](src/main/java/ai/vespa/example/album/MetalSearcher.java)
for an example of tracing in custom Searcher code.


## Custom metrics
See [MetalSearcher](src/main/java/ai/vespa/example/album/MetalSearcher.java)
for an examples of a custom metric - a counter for each successful lookup.
[services.xml](src/main/application/services.xml) has an `admin` section mapping the metric
into a `consumer` that can be used in the [metrics APIs](https://docs.vespa.ai/en/operations/metrics.html).
Also see [MetalSearcherTest](src/test/java/ai/vespa/example/album/MetalSearcherTest.java)
for how to implement unit tests.

Run a query like:

    $ vespa query "select * from music where album contains 'metallica'" searchChain=metalchain

to see the custom metric in
<a href="http://localhost:19092/metrics/v1/values?consumer=my-metrics" data-proofer-ignore>
http://localhost:19092/metrics/v1/values?consumer=my-metrics</a>

This code uses a [Counter](https://github.com/vespa-engine/vespa/blob/master/container-core/src/main/java/com/yahoo/metrics/simple/Counter.java) -
A [Gauge](https://github.com/vespa-engine/vespa/blob/master/container-core/src/main/java/com/yahoo/metrics/simple/Gauge.java)
example, with a dimension could be like:

````
public class HitCountSearcher extends Searcher {
    private static final String LANGUAGE_DIMENSION_NAME = "query_language";
    private static final String EXAMPLE_METRIC_NAME = "example_hitcounts";
    private final Gauge hitCountMetric;

    public HitCountSearcher(MetricReceiver receiver) {
        this.hitCountMetric = receiver.declareGauge(EXAMPLE_METRIC_NAME, Optional.empty(),
                new MetricSettings.Builder().build());
    }

    @Override
    public Result search(Query query, Execution execution) {
        Result result = execution.search(query);
        hitCountMetric
                .sample(result.getTotalHitCount(),
                        hitCountMetric.builder()
                                .set(LANGUAGE_DIMENSION_NAME, query.getModel().getParsingLanguage().languageCode())
                                .build());
        return result;
    }
}
````

Also see [histograms](https://docs.vespa.ai/en/operations-selfhosted/monitoring.html#histograms).
