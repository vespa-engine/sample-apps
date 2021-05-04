<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa sample applications - album recommendations docproc

Data written to Vespa pass through document processing,
where [indexing](https://docs.vespa.ai/en/indexing.html) is one example.

Applications can add custom processing, normally done before indexing.
This is done by adding a [Document Processor](https://docs.vespa.ai/en/document-processing.html).
Such processing is synchronous, and this is problematic for processing
that requires other resources with high latency -
this can saturate the threadpool.

This application demonstrates how to use _Progress.LATER_
and the asynchronous [Document API](https://docs.vespa.ai/en/document-api-guide.html)

Summary:
- Document Processors: modify / enrich data in the feed pipeline
- Multiple Schemas: store different kinds of data, like different database tables
- Enrich data from multiple sources: here, look up data in one schema and add to another
- Document API: write asynchronous code to fetch data

Flow:
![image](img/async-docproc.svg)
1. Feed album document with _music_ schema
1. Look up in _lyrics_ schema if album with given ID has lyrics stored
1. Store album with lyrics in _music_ schema 


Create an application in the Vespa Cloud.
Log in to [console.vespa.ai](http://console.vespa.ai) and click "Create application", choose a name like _album-rec-docproc_.
This requires a Google or GitHub account, and will start your free trial if you don't already have a Vespa Cloud tenant.

Clone sample apps:
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/vespa-cloud/album-recommendation-docproc
</pre>


Create a self-signed certificate. This certificate and key will be used to send requests to Vespa Cloud.
<pre data-test="exec">
$ openssl req -x509 -nodes -days 14 -newkey rsa:4096 \
-subj "/CN=cloud.vespa.example" \
-keyout data-plane-private-key.pem -out data-plane-public-cert.pem
$ mkdir -p src/main/application/security
$ cp data-plane-public-cert.pem src/main/application/security/clients.pem
</pre>


Create a deployment API key. In [console.vespa.ai](http://console.vespa.ai),
choose tenant and click _Keys_ to generate and save the _user API key_.
The key is saved to `$HOME/Downloads/USER.TENANTNAME.pem.


Set the tenant and application name.
Update `pom.xml` with the `tenant` and `application` names you chose when creating the application in the console.


Build and deploy the application. This deploys an instance (with a name you choose here) of the application to the `dev` zone:
<pre>
$ mvn package vespa:deploy -DapiKeyFile=$HOME/Downloads/USER.TENANTNAME.pem -Dinstance=my-instance
</pre>
<!-- Version of the above for automatic testing -->
<pre data-test="exec" style="display:none">
$ API_KEY=`echo $VESPA_TEAM_API_KEY | openssl base64 -A -a -d`
$ mvn clean package vespa:deploy -DapiKey="$API_KEY" -Dinstance=my-instance
</pre>
The first deployment can take a few minutes.


Verify that you can reach the application endpoint.
The endpoint URL is printed in the _Install application_ section when the deployment is successful.
Put this in an environment variable and verify it.
<pre data-test="exec">
$ ENDPOINT=https://my-instance.album-rec-docproc.vespa-team.aws-us-east-1c.dev.public.vespa.oath.cloud
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem $ENDPOINT
</pre>
You can also [do this in a browser](https://cloud.vespa.ai/en/security-model#using-a-browser).


Feed a _lyrics_ document:
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    -H "Content-Type:application/json" --data-binary @src/test/resources/A-Head-Full-of-Dreams-lyrics.json \
    $ENDPOINT/document/v1/mynamespace/lyrics/docid/1
</pre>

Get the document to validate - dump all docs in lyrics schema:
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    "$ENDPOINT/document/v1/mynamespace/lyrics/docid?wantedDocumentCount=100"
</pre>

Feed a _music_ document:
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    -H "Content-Type:application/json" --data-binary @src/test/resources/A-Head-Full-of-Dreams.json \
    $ENDPOINT/document/v1/mynamespace/music/docid/1
</pre>

Get the document to validate - dump all docs in music schema - see lyrics in music document:
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    "$ENDPOINT/document/v1/mynamespace/music/docid?wantedDocumentCount=100"
</pre>

Use the https://console.vespa.oath.cloud to download logs, then inspect what happened:
```
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	In process
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Added to requests pending: 1
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Request pending ID: 1, Progress.LATER
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	In process
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Request pending ID: 1, Progress.LATER
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	In handleResponse
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Async response to put or get, requestID: 1
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Found lyrics for : document 'id:mynamespace:lyrics::1' of type 'lyrics'
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	In process
Container.ai.vespa.example.album.LyricsDocumentProcessor	info	  Set lyrics, Progress.DONE
```
- In the first invocation of _process_, an async request is made - set _Progress.LATER_
- In the second invocation of _process_, the async request has not yet completed
  (there can be many such invocations)  - set _Progress.LATER_
- Then, the handler for the async operation is invoked as the call has completed
- In the subsequent _process_ invocation, we see that the async operation has completed -
  set _Progress.DONE_

Get / delete a document by ID:
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    "$ENDPOINT/document/v1/mynamespace/music/docid/1"
    
$ curl -X DELETE --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    "$ENDPOINT/document/v1/mynamespace/music/docid/1"
</pre>
