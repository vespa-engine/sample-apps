<!-- Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->
# Vespa basic search example on Docker Swarm 

Please refer to
[Vespa quick start using Docker](http://docs.vespa.ai/documentation/vespa-quick-start.html)
for more information on the basic single container example.

This example assumes that a [Docker Swarm](https://docs.docker.com/engine/swarm/) is up and running and that you have console access to one of the masters. 

[Play with Docker](https://labs.play-with-docker.com) is a free service that will provide a Docker Swarm cluster if you do not have access to one. To create 
Swarm, hit the green start button, click on the wrench in the top left and choose one of the templates. This should give you 5 nodes.

The example below needs to be executed on one of the master nodes.

### Executable example
**Check-out the example repository:**
<pre data-test="exec">
$ git clone https://github.com/vespa-engine/sample-apps.git
$ export VESPA_SAMPLE_APP=`pwd`/sample-apps/basic-search-on-docker-swarm
$ cd `pwd`/sample-apps && git checkout aressem/add-multicontainer-example
</pre>
**Deploy the Vespa stack:**
<pre data-test="exec">
$ docker stack deploy -c $VESPA_SAMPLE_APP/docker-compose.yml vespa
</pre>
**Wait for successful deployment of the stack (the REPLICAS column should show N/N for the Vespa services):**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/wait_until_all_stack_services_running.sh
</pre>
**Generate the hosts.xml file based on running containers:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/generate_hosts_xml.sh | tee $VESPA_SAMPLE_APP/src/main/application/hosts.xml 
</pre>
**Wait for the configuration server to start (should return 200 OK):**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head $(hostname):19071/ApplicationStatus
</pre>
**Deploy the application:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/deploy.sh
</pre>
**Wait for the application to start (should return 200 OK):**
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://$(hostname):8080/ApplicationStatus
</pre>
**Feed data to the application:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/feed.sh
</pre>
**Do a search:**
<pre data-test="exec">
$ curl -s "http://$(hostname):8080/search/?query=bad" | python -m json.tool
</pre>
**Remove the stack:**
<pre data-test="after">
$ docker stack rm vespa
</pre>
