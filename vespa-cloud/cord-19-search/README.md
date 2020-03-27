<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - CORD-19 

Vespa application which index the [CORD-19](https://pages.semanticscholar.org/coronavirus-research) dataset.

* Frontend:  
* Full API access: 


## API Access  
For using the Search Api of Vespa please see  [API documentation](https://docs.vespa.ai/documentation/search-api.html), [YQL Query Language](https://docs.vespa.ai/documentation/query-language.html).
For the full document definition see [doc.sd](src/main/application/searchdefinition/doc.sd).

### High level field description 
These are the most important fields in the dataset

|field|source in CORD-19|indexed/searchable|summary (returned with hit)|available for grouping|matching|Vespa type|
|---|---|---|---|---|--|--|
|default|title + abstract|yes|no|no|tokenized and stemmed (match:text)|fieldset |
|all |title + abstract + body_text|yes|no|no|tokenized and stemmed (match:text)|fieldset |
|title|title from metadata or from contents of *sha* json file|yes|yes with bolding|no|tokenized and stemmed (match:text)|string|
|abstract|abstract|yes|yes with bolding and dynamic summary|no|tokenized and stemmed (match:text)|string|
|body_text|All body_text sections|yes|yes with bolding and dynamic summary|no|tokenized and stemmed (match:text)|string|
|datestring|datestring from metadata|no|yes|yes|no|string|
|timestamp|Epoch Unix time stamp parsed from datestring|yes|yes|yes|range and exact matching - can also be sorted on|long|
|license|license|yes|yes|yes|exact matching|string|
|has_full_text|has_full_text|yes|yes|yes|exact matching|bool|
|doi|https:// + doi from metadata|no|yes|no|no|bool|
|id|row id from metadata.csv|yes|yes|yes|yes|int|
|title_embedding|[SciBERT-NLI](https://huggingface.co/gsarti/scibert-nli) embedding from title|yes (using nearestNeighbor())|no|no|yes|tensor<float>(x[768])|
|abstract_embedding|[SciBERT-NLI](https://huggingface.co/gsarti/scibert-nli) embedding from abstract|yes (using nearestNeighbor())|no|no|yes|tensor<float>(x[768])|
|authors|authors for full documents|yes using sameElement()|yes|yes|yes|array of struct|


## Ranking
See Vespa's [Ranking documentation](https://docs.vespa.ai/documentation/ranking.html). There are 3 ranking profiles available (&ranking.profile=x).:

|Ranking|Description|
|---|---|
|default|The default Vespa ranking function (nativeRank) which also uses term proximity for multi-term queries|
|bm25|A weighted combination of bm25(title), bm25(abstract) and bm25(body_text)|

The ranking profiles are defined in the [document definition (doc.sd)](src/main/application/searchdefinition/doc.sd).

## Example API queries
For using the Search Api of Vespa please see  [API documentation](https://docs.vespa.ai/documentation/search-api.html), [YQL Query Language](https://docs.vespa.ai/documentation/query-language.html).
In the below examples we use python with the requests api, using the POST search api.
```python
import requests 

#Search for documents matching all query terms (either in title or abstract)
search_request_all = {
  'yql': 'select id,title, abstract, doi from sources * where userQuery();',
  'hits': 5,
  'summary': 'short',
  'timeout': '1.0s',
  'query': 'coronavirus temperature sensitivity',
  'type': 'all',
  'ranking': 'default'
}

#Search for documents matching any of query terms (either in title or abstract)
search_request_any = {
  'yql': 'select id,title, abstract, doi from sources * where userQuery();',
  'hits': 5,
  'summary': 'short',
  'timeout': '1.0s',
  'query': 'coronavirus temperature sensitivity',
  'type': 'any',
  'ranking': 'default'
}

#Restrict matching to abstract field 
search_request_all_abstract = {
  'yql': 'select id,title, abstract, doi from sources * where userQuery() and has_full_text=true and timestamp > 1577836800;',
  'default-index': 'abstract',
  'hits': 5,
  'summary': 'short',
  'timeout': '1.0s',
  'query': '"sars-cov-2" temperature',
  'type': 'all',
  'ranking': 'default'
}

#Search authors which is an array of struct using sameElement operator
search_request_authors= {
  'yql': 'select id,authors from sources * where authors contains sameElement(first contains "Keith", last contains "Mansfield");',
  'hits': 5,
  'summary': 'short',
  'timeout': '1.0s',
}

#Sample request 
endpoint='https://api.cord19.vespa.ai/default/search/'
response = requests.post(endpoint, json=search_request_all)
```
