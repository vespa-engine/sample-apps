<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - album recommendations docproc

Data written to Vespa pass through document processing,
where [indexing](https://docs.vespa.ai/en/indexing.html) is one example.
Applications can add custom processing, normally done before indexing.
This is done by adding a [Document Processor](https://docs.vespa.ai/en/document-processing.html).
Such processing is synchronous, and this is problematic for processing
that requires other resources with high latency -
this can saturate the threadpool.

This application demonstrates how to use _Progress.LATER_
and the asynchronous [Document API](https://docs.vespa.ai/en/document-api-guide.html). Summary:
- Document Processors: modify / enrich data in the feed pipeline
- Multiple Schemas: store different kinds of data, like different database tables
- Enrich data from multiple sources: here, look up data in one schema and add to another
- Document API: write asynchronous code to fetch data

Flow:
1. Feed album document with the _music_ schema
1. Look up in the _lyrics_ schema if album with given ID has lyrics stored
1. Store album with lyrics in the _music_ schema

![image](img/async-docproc.svg)


## Steps

**Create a Vespa Cloud application:**
Use _Create application_ at [console.vespa.ai](http://console.vespa.ai),
choose a name like _album-rec-docproc_.
This requires a Google or GitHub account,
and will start your free trial if you don't already have a Vespa Cloud tenant.


**Clone sample apps:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/vespa-cloud/album-recommendation-docproc
</pre>


**Create a self-signed certificate:**
This certificate/key pair will be used to access the application deployed to Vespa Cloud.
<pre data-test="exec">
$ openssl req -x509 -nodes -days 14 -newkey rsa:4096 \
-subj "/CN=cloud.vespa.example" \
-keyout data-plane-private-key.pem -out data-plane-public-cert.pem
$ mkdir -p src/main/application/security
$ cp data-plane-public-cert.pem src/main/application/security/clients.pem
</pre>


**Create an API key:**
In [console.vespa.ai](http://console.vespa.ai),
choose _tenant_ and click _Keys_ to generate and save the _user API key_.
The key is saved to `$HOME/Downloads/USER.TENANTNAME.pem`.


**Set tenant and application name in pom.xml:**
Set the `tenant` and `application` names used when creating the application above.


**Build and deploy:**
This builds and deploys an application instance called _my-instance_
to the `dev` [zone](https://cloud.vespa.ai/en/reference/zones).
The first deployment can take a few minutes:
<pre>
$ mvn clean package vespa:deploy -DapiKeyFile=$HOME/Downloads/USER.TENANTNAME.pem -Dinstance=my-instance
</pre>
Alternatively, using a key value from an environment variable;
<pre data-test="exec">
$ API_KEY=`echo $VESPA_TEAM_API_KEY | openssl base64 -A -a -d`
$ mvn clean package vespa:deploy -DapiKey="$API_KEY" -Dinstance=my-instance
</pre>


**Verify endpoint:**
The endpoint URL is output above "Installation succeeded!" in the deployment log.
Put this in an environment variable and verify it
(You can also [do this in a browser](https://cloud.vespa.ai/en/security-model#using-a-browser)):
<pre data-test="exec">
$ ENDPOINT=https://my-instance.album-rec-docproc.vespa-team.aws-us-east-1c.dev.public.vespa.oath.cloud
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem $ENDPOINT
</pre>


**Feed a _lyrics_ document:**
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    -H "Content-Type:application/json" --data-binary @src/test/resources/A-Head-Full-of-Dreams-lyrics.json \
    $ENDPOINT/document/v1/mynamespace/lyrics/docid/1
</pre>
Get the document to validate:
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    $ENDPOINT/document/v1/mynamespace/lyrics/docid/1
</pre>


**Feed a _music_ document:**
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    -H Content-Type:application/json --data-binary @src/test/resources/A-Head-Full-of-Dreams.json \
    $ENDPOINT/document/v1/mynamespace/music/docid/1
</pre>
Get the document to validate - see lyrics in music document:
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    $ENDPOINT/document/v1/mynamespace/music/docid/1
</pre>
Compare, the original document did not have lyrics - it has been added in the
[LyricsDocumentProcessor](src/main/java/ai/vespa/example/album/LyricsDocumentProcessor.java):
<pre data-test="exec">
$ cat src/test/resources/A-Head-Full-of-Dreams.json
</pre>
Use the [console](http://console.vespa.ai) to download logs, then inspect what happened:
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
- In the first invocation of _process_, an async request is made - set `Progress.LATER`
- In the second invocation of _process_, the async request has not yet completed
  (there can be many such invocations)  - set `Progress.LATER`
- Then, the handler for the async operation is invoked as the call has completed
- In the subsequent _process_ invocation, we see that the async operation has completed -
  set `Progress.DONE`


**Other /document/v1 operations:**
Dump documents using [visiting](https://docs.vespa.ai/en/content/visiting.html):
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    $ENDPOINT/document/v1/mynamespace/music/docid?wantedDocumentCount=100
</pre>

Update a field (get it again to see the field change):
<pre data-test="exec">
$ curl --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    -X PUT -H Content-Type:application/json \
    --data '{ "fields": { "album": { "assign": "A Head Half Full of Dreams" } } }' \
    $ENDPOINT/document/v1/mynamespace/music/docid/1
</pre>

Delete a document:
<pre data-test="exec">
$ curl -X DELETE --cert ./data-plane-public-cert.pem --key ./data-plane-private-key.pem \
    $ENDPOINT/document/v1/mynamespace/music/docid/1
</pre>

--------------------------------


## Further reading

* https://docs.vespa.ai/en/getting-started.html
* https://docs.vespa.ai/en/developer-guide.html
