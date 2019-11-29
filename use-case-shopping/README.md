<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Vespa sample applications - e-commerce

A sample application showcasing a simple e-commerce site built with Vespa. Refer to [Use Case - shopping](https://docs.vespa.ai/documentation/use-case-shopping.html).

Included scripts to convert data from Julian McAuley's Amazon product data set
(http://jmcauley.ucsd.edu/data/amazon/links.html)
to a Vespa data feed. This repository contains a small sample of this data from
the sports and outdoor category, but you can download other data from the site
above and use the scripts to convert.


### How to run
Prerequisites: git, Java 11, mvn 3.6.1 and a X.509 certificate.
The certificate is used to access the application's endpoints.
You also need _vespa-http-client-jar-with-dependencies.jar_ - build Vespa to get that <!-- ToDo: FIXME -->

1.  Go to http://console.vespa.ai/, click "Create application"

1.  Download sample apps:
     ```sh
     $ git clone https://github.com/vespa-engine/sample-apps.git && cd sample-apps/use-case-shopping
     ```

1.  Get a X.509 certificate. To create a self-signed certificate
(more details in  in [Data Plane](https://cloud.vespa.ai/security-model.html#data-plane), see _Client certificate_), do
    ```sh
    $ openssl req -x509 -nodes -days 14 -newkey rsa:4096 \
    -subj "/C=NO/ST=Trondheim/L=Trondheim/O=My Company/OU=My Department/CN=example.com" \
    -keyout data-plane-private-key.pem -out data-plane-public-cert.pem
    ```

1.  Edit the properties `tenant` and `application` in `pom.xml` — use the values entered in the console in 2. 

1.  Add certificate to application package
    ```sh
    $ mkdir -p src/main/application/security && cp data-plane-public-cert.pem src/main/application/security/clients.pem
    ```

1.  Build the app:
     ```sh
     $ mvn clean package
     ```

 1.  Deploy with a key pair (see [album-recommendation-java](../album-recommendation-java) for other deploy options):
    1. In the console, navigate to your tenant, and click _Keys_, then generate a random key;
the key is downloaded to
       `$HOME/Downloads/TENANTNAME.pem`.
    1. Set the `apiKeyFile` property in `pom.xml` to the absolute path of the key, **or**
    1. on _each_ `mvn` invocation throughout, specify `-DapiKeyFile=/path/to/key.pem`
    1. Deploy the application to `dev` and wait for it to start (optionally specifying `-DapiKeyFile`)
       ```sh
       $ mvn vespa:deploy
       ```
    1. Now is a good time to read [http://cloud.vespa.ai/automated-deployments](automated-deployments),
    as first time deployments takes a few minutes.
    Seeing CERTIFICATE_NOT_READY / PARENT_HOST_NOT_READY / LOAD_BALANCER_NOT_READY is normal.
    The endpoint URL is output when the deployment is successful - copy this for the next step.

1.  Store the endpoint of the application:
    ```sh
    $ ENDPOINT=https://end.point.name
    ```
    Try the endpoint to validate it is up:
    ```sh
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem $ENDPOINT
    ```

1.  Create data feed:
    First, create data feed for products:
    ```sh
    $ cat meta_sports_20k_sample.json | ./convert_meta.py > feed_items.json
    ```

    Next, data feed for reviews:
    ```sh
    $ cat reviews_sports_24k_sample.json | ./convert_reviews.py > feed_reviews.json
    ```

1.  Feed data:
    Feed products:
    ```sh
    $ java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file feed_items.json --endpoint $ENDPOINT
    ```

    Feed reviews:
    ```sh
    $ java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file feed_reviews.json --endpoint $ENDPOINT
    ```

1.  Test the application:
    ```sh
    $ curl -s http://localhost:8080/search/?yql=select%20%2A%20from%20sources%20%2A%20where%20default%20contains%20%22golf%22%3B
    ```

1.  Browse the site:
    [http://ENDPOINT/site](http://ENDPOINT/site)









----
### How to run

**Check-out, compile and run:**

<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APPS=`pwd`/sample-apps
$ cd $VESPA_SAMPLE_APPS/use-case-shopping &amp;&amp; mvn clean package
$ docker run --detach --name vespa --hostname vespa-container --privileged \
  --volume $VESPA_SAMPLE_APPS:/vespa-sample-apps --publish 8080:8080 vespaengine/vespa
</pre>

**Wait for the configserver to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ docker exec vespa bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
</pre>

**Deploy the application:**

<pre data-test="exec">
$ docker exec vespa bash -c '/opt/vespa/bin/vespa-deploy prepare /vespa-sample-apps/use-case-shopping/target/application.zip && \
  /opt/vespa/bin/vespa-deploy activate'
</pre>

**Wait for the application to start:**

<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8080/ApplicationStatus
</pre>

**Create data feed:**

First, create data feed for products:

<pre data-test="exec">
$ cat meta_sports_20k_sample.json | ./convert_meta.py > feed_items.json
</pre>

Next, data feed for reviews:

<pre data-test="exec">
$ cat reviews_sports_24k_sample.json | ./convert_reviews.py > feed_reviews.json
</pre>

**Feed data:**

Feed products:

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /vespa-sample-apps/use-case-shopping/feed_items.json --host localhost --port 8080'
</pre>

Feed reviews:

<pre data-test="exec">
$ docker exec vespa bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --file /vespa-sample-apps/use-case-shopping/feed_reviews.json --host localhost --port 8080'
</pre>

**Test the application:**

<pre data-test="exec" data-test-assert-contains="id:item:item::">
$ curl -s http://localhost:8080/search/?query=default:golf
</pre>

**Browse the site:**

[http://localhost:8080/site](http://localhost:8080/site)

**Shutdown and remove the container:**
<pre data-test="after">
$ docker rm -f vespa
</pre>




