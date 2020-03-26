<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
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
See Vespa's [Ranking documentation](https://docs.vespa.ai/documentation/ranking.html). There are 3 ranking profiles available (&ranking.profile=x)
There 
|Ranking|Description|
|---|---|
|---|---|


## Example API queries

