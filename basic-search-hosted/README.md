<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Hosted Vespa sample applications — Basic hosted stateless Vespa application

## Getting started
Prerequisites: git, Java 11, mvn 3.6.1

1. Go to https://console.vespa-external.aws.oath.cloud/

2. Click "Create new tenant", then go to this tenant and click "Create application"

3. Download sample app:
 ```sh
 $ git clone git@github.com:vespa-engine/sample-apps.git && cd sample-apps/basic-search-hosted
 ```
 
4. Edit properties _tenant_, _application_ and _instance_ in _pom.xml_ -
use values from the console (what was used to create the application) - use "default" as instance name

5. Build java sample app:
 ```sh
 $ mvn install package
 ```
 
6. In the console, click "Deploy". Generate the deploy key (this downloads the key file),
then use this to deploy to Vespa:
```sh
$ mvn vespa:deploy -DprivateKeyFile=$HOME/Downloads/mytenantname.myappname.myinstancename.pem
```

7. Alternatively, in the "Deploy to dev" console section, upload _target/application.zip_ - click Deploy

8. Click "deployment log" to track the deployment. "Installation succeeded!" in the bottom pane indicates success 

9. Click "Instances" at the top, then "endpoints". Click the endpoint to validate it is up. _Temporary workaround: use http (not https) and port 443) - example http://end.point.name:443_.
One can also use:
```sh
$ mvn -DprivateKeyFile=$HOME/Downloads/mytenantname.myappname.myinstancename.pem vespa:endpoints # test this!
```

10. Feed documents
```sh
$ curl -H "Content-Type:application/json" --data-binary  @music-data-1.json http://endpoint:443/document/v1/music/music/docid/1
$ curl -H "Content-Type:application/json" --data-binary  @music-data-2.json http://endpoint:443/document/v1/music/music/docid/2
```

11. Visit documents
```sh
$ curl http://endpoint:443/document/v1/music/music/docid?wantedDocumentCount=100
```

12. Search documents
```sh
$ curl http://endpoint:443/search/?query=bad
```


## Local development

Included are detailed functional tests for the Vespa application, written in JUnit. While the
main intention for such functional tests is to run them in a continuous
deployment pipeline, <!-- TODO LINK --> to verify changes to the Vespa application before
they are deployed to the production environments, these particular tests also

* demonstrate how to communicate with the Vespa deployment over HTTP,
* serve as introductory documentation for the most central of Vespa's features, and
* provide a starting point for thorough functional tests of your own Vespa application.

The tests require a running Vespa deployment, e.g., a local
Vespa deployment running within docker.

<!-- TODO: Un-comment the below warning -->
<!-- em>This only works with self-hosted `services.xml` and `hosts.xml`, which can be found in any of the other sample apps.</em -->

### Run those JUnit tests against the local docker container
Assuming the below is done, put
<pre>
{
  "clusters": [
    { "container": "http://localhost:8080/" }
  ]
}
</pre>
in some file `/path/to/test/config`, and run JUnit tests with `-Dvespa.test.config=/path/to/test/config -Dtest.categories=system
to run all `@SystemTest` classes; `staging` and `production` selects the other suites of tests, and `integration` all of them.

### Executable example
**Check-out, compile and run:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/basic-search-hosted &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>

**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/basic-search-hosted/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>

**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Test the application:**
<pre data-test="exec" data-test-assert-contains='"totalCount": 0'>
$ curl -s http://localhost:8080/search/?query=name:foo
</pre>

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
