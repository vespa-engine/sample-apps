<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Visit vespa documents 

## Steps

1. `git clone git@github.com:vfil/sample-apps.git`
2. `export VESPA_SAMPLE_APPS=`pwd`/sample-apps`
3. Run vespa sample app:
```
docker run --detach --name vespa --hostname vespa-container --privileged   --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 -p 19070-19899:19070-19899 --hostname localhost vespaengine/vespa
```

4. Wait for the configuration server to start - signified by a 200 OK response:
`docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'`

5. Deploy and activate a sample application:
```
docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/basic-search/src/main/application/ && \
       /opt/vespa/bin/vespa-deploy activate'
```

6. Ensure the application is active - wait for a 200 OK response:
`curl -s --head http://localhost:8080/ApplicationStatus`

7. Feed documents into Vespa:
```
$ curl -s -H "Content-Type:application/json" --data-binary @${VESPA_SAMPLE_APPS}/basic-search/music-data-1.json \
    http://localhost:8080/document/v1/music/music/docid/1 | python -m json.tool
$ curl -s -H "Content-Type:application/json" --data-binary @${VESPA_SAMPLE_APPS}/basic-search/music-data-2.json \
    http://localhost:8080/document/v1/music/music/docid/2 | python -m json.tool
```

9. Set the configsources var:
`export VESPA_CONFIG_SOURCES=localhost:19090`

8. run Visitor main:
`mvn exec:java -Dexec.mainClass="ai.vespa.example.visit.Visit"`

