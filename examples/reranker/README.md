<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Reranker sample application

A stateless application which reranks results obtained from another Vespa application.
While this does not result in good performance and is not recommended for production, 
it is useful when you want to quickly do ranking experiments without rewriting application data.

## Usage

1. Make sure the application to rerank has a 
[token endpoint](https://cloud.vespa.ai/en/security/guide#application-key).
2. `vespa clone examples/reranker`
3. Add the endpoint and any defaults to the reranker config in `src/main/application/services.xml`
   (parameters can also be passed in the request).
4. Add the model(s) to use for reranking to the `models` directory.
5. `mvn install && vespa deploy`
6. Issue queries. All request parameters including the token header will be passed through to the application to be reranked.

Example requests:

Minimal:

    vespa query "select * from sources * where album contains 'to'" --header "Authorization: Bearer [your token]"

Passing all reranking parameters:

    vespa query "select * from sources * where album contains 'to'" --header "Authorization: Bearer [your token]" rerank.model=xgboost_model_example rerank.hits=100 profile=firstPhase
