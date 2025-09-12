<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.-->

# Using query profiles for different use cases

Vespa supports [query-profiles](https://docs.vespa.ai/en/query-profiles.html?mode=selfhosted#using-a-query-profile), which let you
define sets of query parameters for different use cases, so that the client
only need to choose the query profile rather than sending long lists of parameters to get the behavior they need.

## Provided query profiles

For this sample app, we have provided 6 query profiles:

1. [`hybrid`](app/search/query-profiles/hybrid.xml)
2. [`rag`](app/search/query-profiles/rag.xml)
3. [`deepresearch`](app/search/query-profiles/deepresearch.xml)
4. [`hybrid-with-gbdt`](app/search/query-profiles/hybrid-with-gbdt.xml)
5. [`rag-with-gbdt`](app/search/query-profiles/rag-with-gbdt.xml)
6. [`deepresearch-with-gbdt`](app/search/query-profiles/deepresearch-with-gbdt.xml)

Each of the query profiles have different query parameters set, such as the search chain to use,
the rank profile, and the number of hits to return.

## Testing the query profiles

Run the command below to use the `rag-with-gbdt` query profile.
Note that the `X-LLM-API-KEY` header is required for this query profile.

<pre>
$ vespa query \
    --header="X-LLM-API-KEY:<your-api-key>" \
    query="Summarize the key architectural decisions documented for SynapseFlow's v0.2 release." \
    queryProfile=rag-with-gbdt
</pre>

Run the command below to use the `deepresearch-with-gbdt` query profile.
<pre data-test="exec" data-test-assert-contains="Architecture Document">
$ vespa query \
    query="Summarize the key architectural decisions documented for SynapseFlow's v0.2 release." \
    queryProfile=deepresearch-with-gbdt
</pre>

