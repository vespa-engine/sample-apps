<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - Basic stateless Vespa application

Extends the [album-recommedations](../album-recommedations) sample application with a Searcher component in Java
which does query and result processing.

Refer to
[developing searchers](http://docs.vespa.ai/documentation/searcher-development.html)
for more information.

See [getting started](http://cloud.vespa.ai/getting-started.html) for troubleshooting.


## Getting started
Prerequisites: git, Java 11, mvn 3.6.1 and a X.509 certificate.
The certificate is used to access the application's endpoints.

1.  Go to http://console.vespa.ai/, click "Create application"

1.  Download sample apps:
     ```sh
     $ git clone https://github.com/vespa-engine/sample-apps.git && cd sample-apps/album-recommendation-java
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
 
1.  Deploy with a key pair (recommended):
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
    The endpoint URL is printed in the _Install application_ section when the deployment is successful -
    copy this for the next step.

1.  ...**or**, deploy through the console:
    1. In the "Deploy to dev" console section, upload _target/application.zip_, then click "Deploy".
    1.  Click "deployment log" to track the deployment. "Installation succeeded!" in the bottom pane indicates success. Click "Zones" at the top, then "endpoints", to find the endpoint URL.

1.  Store the endpoint of the application:
    ```sh
    $ ENDPOINT=https://end.point.name
    ```
    Try the endpoint to validate it is up:
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

1.  [https://docs.vespa.ai/documentation/content/visiting.html](Visit) documents:
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

1.  If you downloaded the deploy key, run the included integration test _ExampleSystemTest_ with
    ```sh
    $ mvn test -Dtest.categories=system
    ```
    or run it directly from the IDE. 
    ai.vespa.hosted.cd.Endpoint must have access to the data plane key and certificate pair,
    to talk to the application endpoint.
    Set these with the `dataPlaneCertificateFile` and `dataPlaneKeyFile` properties,
    in the same manner as the `apiKeyFile`:
    -   in `pom.xml`, like `<dataPlaneCertificateFile>data-plane-public-cert.pem</dataPlaneCertificateFile>`,
    -   as arguments to `mvn` like `-DdataPlaneCertificateFile=data-plane-public-cert.pem`, or
    -   as VM options for the JUnit tests in the IDEA, like `-DdataPlaneCertificateFile=data-plane-public-cert.pem`. 
    This also applies to the `dataPlaneKeyFile`, and the `apiKeyFile`.


<!-- Troubleshooting notes
* if the bundle name is changed, it can cause container not to start and deploy fail - hard to get to logs then ...
*
-->
