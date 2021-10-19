<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - document processing

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
2. Look up in the _lyrics_ schema if album with given ID has lyrics stored
3. Store album with lyrics in the _music_ schema

![image](img/async-docproc.svg)



## Steps

### Install the Vespa CLI:
Using [Homebrew](https://brew.sh/):
```
$ brew install vespa-cli
```
You can also [download Vespa CLI](https://github.com/vespa-engine/vespa/releases)
for Windows, Linux and macOS.



###Create your application in the Vespa Cloud:
If you don't already have a Vespa Cloud tenant,
create one at [console.vespa.ai](http://console.vespa.ai).
This requires a Google or GitHub account, and will start your [free trial](https://cloud.vespa.ai/pricing#free-trial).



<!-- Use a distinct app name to avoid key collisions -->
### Initialize `myapp-docproc/`
Initialize `myapp-docproc/` to a copy of a
[sample application package](https://docs.vespa.ai/en/cloudconfig/application-packages.html):
<pre data-test="exec">
$ vespa clone vespa-cloud/document-processing myapp-docproc
$ cd myapp-docproc
</pre>



### Let pom.xml specify the right tenant and application name:
Set the properties `tenant` and `application` properties in pom.xml
to your tenant and application name ("myapp-docproc")



### Tell the Vespa CLI to use Vespa Cloud, with your application:
```
$ vespa config set target cloud
$ vespa config set application <tenant-name>.myapp-docproc.default
```
Use the tenant and application name from step 2.

<!-- Override VESPA_CLI_HOME to work around container filesystem limitations and set application used for automatic testing -->
<!-- $PWD is set to $SD_DIND_SHARE_PATH by screwdriver.yaml - a special Docker-in-Docker path -->
<pre data-test="exec" style="display:none">
$ export VESPA_CLI_HOME=$PWD/.vespa TMPDIR=$PWD/.tmp
$ mkdir -p $TMPDIR
$ vespa config set target cloud
$ vespa config set application vespa-team.document-processing.my-instance
</pre>



### Create a user API key:
```
$ vespa api-key
```
Follow the instructions from the command to register the key.

<!-- Write API key used in automatic testing -->
<!-- $VESPA_TEAM_API_KEY is a base64-encoded PEM-encoded EC private key in PKCS#8 format. -->
<!-- However, the current version of Vespa CLI only supports PEM-encoded raw EC private keys, so we have to convert it -->
<pre data-test="exec" style="display:none">
$ echo "$VESPA_TEAM_API_KEY" | openssl base64 -A -a -d | openssl ec > $VESPA_CLI_HOME/vespa-team.api-key.pem
</pre>



### Create a self-signed certificate for accessing your application:
<pre data-test="exec">
$ vespa cert
</pre>

See the [security model](security-model#data-plane) for more details.



### Build and deploy the application:
```
$ mvn package vespa:deploy
```
The first deployment may take a few minutes.

<!-- Set instance used in automatic testing -->
<pre data-test="exec" style="display:none">
$ mvn package vespa:deploy -Dinstance=my-instance -DapiKeyFile=$VESPA_CLI_HOME/vespa-team.api-key.pem
</pre>



### Verify that you can reach your application endpoint:
<pre data-test="exec">
$ vespa status --wait 300
</pre>



### Feed a _lyrics_ document:
... and get the document after the feed as well:

<pre data-test="exec">
$ vespa document src/test/resources/A-Head-Full-of-Dreams-lyrics.json
$ vespa document get id:mynamespace:lyrics::a-head-full-of-dreams
</pre>



### Feed a _music_ document:
Get the document to validate - see lyrics in music document:

<pre data-test="exec">
$ vespa document src/test/resources/A-Head-Full-of-Dreams.json
$ vespa document get id:mynamespace:music::a-head-full-of-dreams
</pre>

Compare, the original document did not have lyrics -
it has been added in the [LyricsDocumentProcessor](src/main/java/ai/vespa/example/album/LyricsDocumentProcessor.java):

<pre>
$ cat src/test/resources/A-Head-Full-of-Dreams.json
</pre>



### Review logs:
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
In the first invocation of process, an async request is made - set `Progress.LATER`
In the second invocation of process, the async request has not yet completed (there can be many such invocations) -
set `Progress.LATER`
Then, the handler for the async operation is invoked as the call has completed
In the subsequent process invocation, we see that the async operation has completed - set `Progress.DONE`



**Further reading:**

* https://docs.vespa.ai/en/getting-started.html
* https://docs.vespa.ai/en/developer-guide.html
