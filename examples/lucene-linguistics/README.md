<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa LuceneLinguistics Demos

A couple of example of how to get started with the `lucene-linguistics`:

- `minimal`: an absolute minimum to get started;
- `advanced-configuration`: shows the configurability and how to solve common problems. 

## Getting started

For both application package the procedure is the same:
go to the application package directory and play with the following commands:

```shell
# Of course make sure that your Docker daemon is running
# make sure that Vespa CLI is installed
brew install vespa-cli
# Maven must be 3.6+
brew install maven

docker run --rm --detach \
  --name vespa \
  --hostname vespa-container \
  --publish 8080:8080 \
  --publish 19071:19071 \
  --publish 19050:19050 \
  vespaengine/vespa:8.204.11

# To observe the logs from LuceneLinguistics run in a separate terminal
docker logs  vespa -f | grep -i "lucene"

vespa status deploy --wait 300

(mvn clean package && vespa deploy -w 100)

vespa feed src/main/application/ext/document.json
vespa query 'yql=select * from lucene where default contains "dogs"' \
  'model.locale=en'
  
# after this query log entry like this should appear:
[2023-08-02 19:57:12.106] INFO    container        Container.com.yahoo.language.lucene.AnalyzerFactory	Analyzer for language=en is from a list of default language analyzers.
```

The query should return:
```json
{
  "root": {
    "id": "toplevel",
    "relevance": 1.0,
    "fields": {
      "totalCount": 1
    },
    "coverage": {
      "coverage": 100,
      "documents": 1,
      "full": true,
      "nodes": 1,
      "results": 1,
      "resultsFull": 1
    },
    "children": [
      {
        "id": "id:mynamespace:lucene::mydocid",
        "relevance": 0.16343879032006287,
        "source": "content",
        "fields": {
          "sddocname": "lucene",
          "documentid": "id:mynamespace:lucene::mydocid",
          "mytext": "Cats and Dogs"
        }
      }
    ]
  }
}
```

### Observing query rewrites

```shell
vespa query 'yql=select * from lucene where default contains "dogs"' \
  'model.locale=en' \
  'trace.level=2' | jq '.trace.children | last | .children[] | select(.message) | select(.message | test("YQL.*")) | .message'
```
Output
```shell
"YQL+ query parsed: [select * from lucene where default contains \"dog\" timeout 10000]"
```
See that the `dogs` rewritten as `dog`.

Change the `model.locale` to other language, change the query, and observe the analysis differences.

### Observing the indexed tokens

It is possible to explore the tokens in the index.
To do that you can run these commands **inside** the Vespa Docker container.

```shell
# Into the Vespa docker
docker exec -it vespa bash
# Trigger the flushing to the disk
vespa-proton-cmd --local triggerFlush

# Show the posting lists
vespa-index-inspect showpostings \
          --indexdir  /opt/vespa/var/db/vespa/search/cluster.content/n0/documents/lucene/0.ready/index/$(ls /opt/vespa/var/db/vespa/search/cluster.content/n0/documents/lucene/0.ready/index/)/ \
          --field mytext --transpose
# =>
# docId = 1
# field = 0 "mytext"
#  element = 0, elementLen = 2, elementWeight = 1
#   pos = 0, word = "cat"
#   pos = 1, word = "dog"

# Show the tokens
vespa-index-inspect dumpwords \
          --indexdir  /opt/vespa/var/db/vespa/search/cluster.content/n0/documents/lucene/0.ready/index/$(ls /opt/vespa/var/db/vespa/search/cluster.content/n0/documents/lucene/0.ready/index/)/ \
          --wordnum \
          --field mytext
# =>
# 1	cat	1
# 2	dog	1
```
