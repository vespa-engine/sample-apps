<!-- Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Secure Vespa with mutually authenticated TLS

Please refer to
[Vespa quick start using Docker](https://docs.vespa.ai/en/vespa-quick-start.html)
for more information on the basic single container example.

This example assumes that a [Docker Swarm](https://docs.docker.com/engine/swarm/) is up and running and that you have console access to one of the masters.

<a href="https://labs.play-with-docker.com" data-proofer-ignore>Play with Docker</a>
is a free service that will provide a Docker Swarm cluster if you do not have access to one. To create
Swarm, hit the green start button, click on the wrench in the top left and choose one of the templates. This should give you 5 nodes.

The example below needs to be executed on one of the master nodes.

### Executable example

**Validate environment, Docker containers shoould have minimum 6G:**
<pre>
$ docker info | grep "Total Memory"
</pre>
**Check-out the example repository:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APP=`pwd`/sample-apps/secure-vespa-with-mtls
</pre>
**Generate keys and certificate chains:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/generate-cert-chains.sh
</pre>
**Deploy the Vespa stack:**
<pre data-test="exec">
$ docker stack deploy --orchestrator swarm --compose-file $VESPA_SAMPLE_APP/docker-compose.yml vespa
</pre>
**Wait for successful deployment of the stack:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/wait_until_all_stack_services_running.sh
</pre>
**Generate the hosts.xml file based on running containers:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/generate_hosts_xml.sh | tee $VESPA_SAMPLE_APP/src/main/application/hosts.xml
</pre>
**Wait for the configuration server to start (should return 200 OK):**
<pre data-test="exec" data-test-wait-for="200 OK">
$ $VESPA_SAMPLE_APP/scripts/vespa-curl.sh -s --head https://localhost:19071/ApplicationStatus
</pre>
**Deploy the application:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/deploy.sh
</pre>
**Wait for the application to start (should return 200 OK):**
<pre data-test="exec" data-test-wait-for="200 OK">
$ $VESPA_SAMPLE_APP/scripts/client-curl.sh -s --head https://localhost:8443/ApplicationStatus
</pre>
**Feed data to the application:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/feed.sh
</pre>
**Do a search:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/client-curl.sh -s "https://localhost:8443/search/?query=michael" | python -m json.tool
</pre>
**Congratulations. You have now deployed and tested a secure Vespa installation.**
**After you have finished testing the Vespa application excute the following step to delete the services:**
<pre data-test="after">
$ docker stack rm vespa
</pre>
