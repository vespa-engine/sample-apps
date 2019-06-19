<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Hosted Vespa sample applications - Basic stateless Vespa application

## Sign up in hosted Vespa console, and create an application with the wanted name

## Configure pom.xml for your hosted Vespa application

* Set the `tenant` and `application` properties in `pom.xml`.

## Generate and upload key pair
Install `openssl` and run
<pre>
openssl ecparam -name prime256v1 -genkey -noout -out private_key.pem
openssl ec -pubout -in private_key.pem -out public_key.pem
</pre>
to generate a private and public key. Upload the public key to the hosted Vespa API
and set the path to the private key in the `pom.xml` properties.

## Set up a CI job which deploys your application
Command to build and submit application to the hosted Vespa API is
<pre>
mvn -P fat-test-application -DauthorEmail=<span style="background-color: yellow;">user@domain</span> clean package vespa:submit 
</pre>

## Deploy to dev and test against it
Command to build and deploy application to the hosted development environment is
<pre>
mvn clean package vespa:deploy 
</pre>
Example System, Staging and Production tests can then be run from the IDEA.



## Local development

Extends the basic-search sample applicationn with a Searcher component in Java
which does query and result processing.

Please refer to
[developing searchers](http://docs.vespa.ai/documentation/searcher-development.html)
for more information.


### Executable example
**Check-out, compile and run:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/basic-search-java &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>
**Wait for the configserver to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>
**Deploy the application:**
<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/basic-search-java/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>
**Wait for the application to start:**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>
**Test the application:**
<pre data-test="exec" data-test-assert-contains="test:hit">
$ curl -s http://localhost:8080/search/?query=title:foo
</pre>
**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>
