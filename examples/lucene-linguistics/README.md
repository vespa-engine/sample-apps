<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa LuceneLinguistics Demos

A couple of example of how to get started with the `lucene-linguistics`:

- `non-java`: an absolute minimum to get started; 
  - TODO: update the bundle to the newest version when released
- `minimal`: minimal Java based project using Lucene Linguistics;
  - TODO: use non snapshot build
- `advanced-configuration`: demonstrates the configurability;
  - TODO: use non snapshot build
- `going-crazy`: demonstrates the advanced setup;
  - TODO: use non snapshot build

## Getting started

For all application packages the procedure is the same:
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
  vespaengine/vespa:8.224.19

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

It is possible to explore the tokens directly in the index.
To do that you can run these commands **inside** the running Vespa Docker container.

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

Have fun!

## Common Issues

The `lucene-linguistics` component is highly configurable.
It has an optional `configDir` configuration parameter of type `path`.
`configDir` is a directory to store linguistics resources, e.g. dictionaries with stopwords, etc., and is relative to the VAP root directory.

There are several known problems that might happen when `configDir` is misconfigured.

### `configDir` is specified but doesn't exist

If the `configDir` doesn't exist then `vespa deploy` would fail with such error:

```shell
Uploading application package ... failed
Error: invalid application package (400 Bad Request)
Invalid application:
Unable to send file specified in com.yahoo.language.lucene.lucene-analysis:
/opt/vespa/var/db/vespa/config_server/serverdb/tenants/default/sessions/4/lucene (No such file or directory)
```

### Empty directory can't be referred

If the `configDir` is set with `foo` which is empty then during deployment you get a misleading error message:
```shell
Uploading application package ... failed
Error: invalid application package (400 Bad Request)
Invalid application:
Unable to send file specified in com.yahoo.language.lucene.lucene-analysis:
/opt/vespa/var/db/vespa/config_server/serverdb/tenants/default/sessions/8/foo (No such file or directory)
```

### Application package root cannot be used as `configDir`

If you try to be clever and set `<configDir>.</configDir>` then application package would be deployed(!) BUT
not converge with the following error:
```shell
Uploading application package ... done

Success: Deployed target/application.zip
WARNING Jar file 'vespa-lucene-linguistics-poc-0.0.1-deploy.jar' uses non-public Vespa APIs: [com.yahoo.language.simple]

Waiting up to 1m40s for query service to become available ...
Error: service 'query' is unavailable: services have not converged
```

And Vespa logs would be filled with such warnings:
```shell
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	Exception in thread "Rpc executorpool-6-thread-5" java.lang.RuntimeException: More than one file reference found for file 'fbcf5c3dc81d9540'
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.filedistribution.FileDownloader.getFileFromFileSystem(FileDownloader.java:109)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.filedistribution.FileDownloader.getFileFromFileSystem(FileDownloader.java:100)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.filedistribution.FileDownloader.getFutureFile(FileDownloader.java:80)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.filedistribution.FileDownloader.getFile(FileDownloader.java:70)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.config.proxy.filedistribution.FileDistributionRpcServer.downloadFile(FileDistributionRpcServer.java:109)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat com.yahoo.vespa.config.proxy.filedistribution.FileDistributionRpcServer.lambda$getFile$0(FileDistributionRpcServer.java:84)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1136)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:635)
[2023-08-02 20:30:47.675] WARNING configproxy      stderr	\tat java.base/java.lang.Thread.run(Thread.java:833)
```

### Harmless warning
`vespa deploy` always warns with:
```shell
WARNING Jar file 'vespa-lucene-linguistics-poc-0.0.1-deploy.jar' uses non-public Vespa APIs: [com.yahoo.language.simple]
```
You can ignore this warning.
