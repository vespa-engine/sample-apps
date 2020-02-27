<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - album recommendations java

Extends the [album-recommendations](../album-recommendations) sample application with a Searcher component in Java
which does query and result processing. Refer to
[developing searchers](http://docs.vespa.ai/documentation/searcher-development.html) for more information.

This sample app introduces how to write system tests and how to integrate with a CI/CD pipeline.

See [getting started](http://cloud.vespa.ai/getting-started.html) for troubleshooting.

Security notes:
*   To deploy to your instance, a _personal API key_ is required.
    See step 4 below.
*   To read and write to the instance's endpoint, a certificate and a private key is required.
    See step 2 below.
    Find more details in [Data Plane](https://cloud.vespa.ai/security-model.html#data-plane), see _Client certificate_.
*   Instead of using a _personal API key_, one can also deploy using the console, see step 5 in
    [album-recommendation](../album-recommendation/README.md).



## Getting started
Prerequisites: git, Java 11, mvn 3.6.1 and openssl.

1.  Download sample apps:
    ```sh
    $ git clone https://github.com/vespa-engine/sample-apps.git && cd sample-apps/album-recommendation-java
    ```

1.  Get a X.509 certificate and private key. Create a self-signed certificate / private key:
    ```sh
    $ openssl req -x509 -nodes -days 14 -newkey rsa:4096 \
    -subj "/C=NO/ST=Trondheim/L=Trondheim/O=My Company/OU=My Department/CN=example.com" \
    -keyout data-plane-private-key.pem -out data-plane-public-cert.pem
    ```

1.  Add certificate to application package (it must be copied as _clients.pem_):
    ```sh
    $ mkdir -p src/main/application/security && cp data-plane-public-cert.pem src/main/application/security/clients.pem
    ```

1.  Go to http://console.vespa.ai/, choose tenant and click _Keys_ to generate and save the _personal API key_.
    The key is saved to `$HOME/Downloads/TENANTNAME.pem`.
    Then click "Create application"

1.  Edit the properties `tenant` and `application` in `pom.xml` — use the values entered in the console in 4.
 
1.  Build the app and deploy it to the `dev` environment and wait for it to start -
    update the `apiKeyFile` to be the file you downloaded above:
    ```sh
    $ mvn clean package vespa:deploy -DapiKeyFile=$HOME/Downloads/TENANTNAME.pem
    ```

1.  Now is a good time to read [http://cloud.vespa.ai/automated-deployments](automated-deployments),
    as first time deployments takes a few minutes.
    Seeing CERTIFICATE_NOT_READY / PARENT_HOST_NOT_READY / LOAD_BALANCER_NOT_READY is normal.
    The endpoint URL is printed in the _Install application_ section when the deployment is successful -
    copy this for the next step.

1.  Store the endpoint of the application:
    ```sh
    $ ENDPOINT=https://end.point.name
    ```
    Try the endpoint:
    ```sh
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem $ENDPOINT
    ```

1.  Feed documents:
    ```sh
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      -H "Content-Type:application/json" --data-binary @src/test/resources/A-Head-Full-of-Dreams.json \
      $ENDPOINT/document/v1/mynamespace/music/docid/1
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      -H "Content-Type:application/json" --data-binary @src/test/resources/Love-Is-Here-To-Stay.json \
      $ENDPOINT/document/v1/mynamespace/music/docid/2
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      -H "Content-Type:application/json" --data-binary @src/test/resources/Hardwired...To-Self-Destruct.json \
      $ENDPOINT/document/v1/mynamespace/music/docid/3
    ```

1.  [Visit](https://docs.vespa.ai/documentation/content/visiting.html) documents:
    ```sh
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      "$ENDPOINT/document/v1/mynamespace/music/docid?wantedDocumentCount=100"
    ```
    
1.  Recommend albums, send user profile in query:
    ```sh
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      "$ENDPOINT/search/?ranking=rank_albums&yql=select%20%2A%20from%20sources%20%2A%20where%20sddocname%20contains%20%22music%22%3B&ranking.features.query(user_profile)=%7B%7Bcat%3Apop%7D%3A0.8%2C%7Bcat%3Arock%7D%3A0.2%2C%7Bcat%3Ajazz%7D%3A0.1%7D"
    ```
    Limit to albums with the term "to" in title:
    ```sh
    $ curl --cert data-plane-public-cert.pem --key data-plane-private-key.pem \
      "$ENDPOINT/search/?ranking=rank_albums&yql=select%20%2A%20from%20sources%20%2A%20where%20album%20contains%20%22to%22%3B&ranking.features.query(user_profile)=%7B%7Bcat%3Apop%7D%3A0.8%2C%7Bcat%3Arock%7D%3A0.2%2C%7Bcat%3Ajazz%7D%3A0.1%7D"
    ```

1.  At this point, the application is built, unit tested, deployed to a _dev_ instance, fed to and a few test queries have been run.
    Safe deployments depends on automated testing.
    Vespa Cloud has support for running _system_ and _staging_ tests for every change to an application.
    These tests are run as JUnit tests, but use the endpoints of a real deployment of the application.
    When _submitting_ an application to Vespa Cloud, a test instance is set up and tests automatically run using its endpoints.
    To develop system and staging tests, deploy the application to _dev_ (like above) and run tests like
    [_FeedAndSearchSystemTest_](src/test/java/ai/vespa/example/album/FeedAndSearchSystemTest.java):
    ```sh
    $ mvn test -Dtest.categories=system \
      -DdataPlaneKeyFile=data-plane-private-key.pem \
      -DdataPlaneCertificateFile=data-plane-public-cert.pem \
      -DapiKeyFile=$HOME/Downloads/TENANTNAME.pem
    ```
    or run it directly from an IDE. 
    ai.vespa.hosted.cd.Endpoint must have access to the data plane key and certificate pair,
    to talk to the application endpoint.

    Find more details in the [Vespa Cloud API](https://cloud.vespa.ai/reference/vespa-cloud-api.html) and
    [automated-deployments](https://cloud.vespa.ai/automated-deployments).

1.  To run tests against a deployment running in Docker on localhost (instead of using _dev_),
    configure endpoint location:
    ```
    {
        "localEndpoints": {
        "container": "http://localhost:8080/"
        }
    }
    ```
    in some file `/path/to/test/config`, and run JUnit tests with `-Dvespa.test.config=/path/to/test/config -Dtest.categories=system`.
    Refer to [album-recommendation-selfhosted](../album-recommendation-selfhosted) for how to create the application package.

1.  When System and Staging tests are ready, deploy to production.
    Command to build and submit application to the hosted Vespa API is
    ```
    mvn clean package vespa:compileVersion -DapiKeyFile=$HOME/Downloads/TENANTNAME.pem
    mvn -P fat-test-application \
    -Dvespa.compile.version="$(cat target/vespa.compile.version)" \
    -DapiKeyFile=$HOME/Downloads/TENANTNAME.pem \
    package vespa:submit
    ```
