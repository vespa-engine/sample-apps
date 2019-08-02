<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Hosted Vespa sample applications — Basic hosted stateless Vespa application

This application demonstrates how to set up and run a basic hosted Vespa application,
and is intended as an introduction to both Vespa and the hosted Vespa service. <!-- TODO LINK -->  

Included are detailed functional tests for the Vespa application, written in JUnit. These tests

* demonstrate how to communicate with the Vespa deployment over HTTP,
* serve as introductory documentation for the most central of Vespa's features, and
* provide a starting point for thorough functional tests of your own Vespa application.

The tests require a running Vespa deployment, which is easily obtained in hosted Vespa's `dev` environment. <!-- TODO LINK -->
It is also possible to run the tests against a Vespa deployment using, e.g., docker, as in the other sample apps. 

The real intention for the functional tests is to run them in the hosted Vespa continuous deployment pipeline, <!-- TODO LINK -->
to verify changes to the Vespa application before they are deployed to the production environments. 

## Sign up in hosted Vespa console, and create an application with the wanted name

## Generate and upload key pair
Install `openssl` and run
<pre>
openssl ecparam -name prime256v1 -genkey -noout -out private_key.pem
openssl ec -pubout -in private_key.pem -out public_key.pem
</pre>
to generate a private and public key. Then upload the public key through the hosted Vespa dashboard.

## Configure pom.xml for your hosted Vespa application
Set the `tenant`, `application`, and `privateKeyFile` properties in `pom.xml`.

## Deploy to dev and test against it
Command to build and deploy application to the hosted development environment is
<pre>
mvn clean package vespa:deploy 
</pre>
Example System, Staging and Production tests can then be run from an IDE without further setup.
<!-- ... or, add a description for users with older than IntelliJ 2012, and Eclipse ... ??? -->

## Set up a CI job which deploys your application
Command to build and submit application to the hosted Vespa API is
<pre>
mvn vespa:compileVersion # Stores the version to compile against in target/vespa.compile.version
mvn -P fat-test-application \
-Dvespaversion="$(cat target/vespa.compile.version)" \
-DauthorEmail=<span style="{background-color: yellow;}">user@domain</span> \
clean package vespa:submit 
</pre>

## Local development

<em>This only works with self-hosted `services.xml` and `hosts.xml`, which can be found in any of the other sample apps.</em>

### Run those JUnit tests against the local docker container
Assuming the below is done, put
<pre>
{
  "application": "this:is:ignored",
  "zone": "dev.local",
  "system": "publiccd",
  "zoneEndpoints": {
    "dev.local": {
      "container": "http://localhost:8080/"
    }
  }
}
</pre>
in some file `/path/to/test/config`, and run JUnit tests with `-Dvespa.test.config=/path/to/test/config -Psystem-tests`
to run all `@SystemTest` classes. `-Pstaging-tests` and `-Pproduction-tests` runs the other suites of tests. 

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


