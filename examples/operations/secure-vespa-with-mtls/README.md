<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)


# Secure Vespa with mutually authenticated TLS

Refer to [Vespa quick start using Docker](https://docs.vespa.ai/en/vespa-quick-start.html)
for more information on the basic single container example.

This example assumes that a [Docker Swarm](https://docs.docker.com/engine/swarm/) is up and running
and that you have console access to one of the masters.

<a href="https://labs.play-with-docker.com" data-proofer-ignore>Play with Docker</a>
is a free service that will provide a Docker Swarm cluster if you do not have access to one.
To create Swarm, hit the green start button, click on the wrench in the top left and choose one of the templates.
This should give you 5 nodes.
See [Swarm init](#swarm-init-and-network) for details.

The example below must be run on one of the master nodes.


## Executable example

**Validate environment, Docker containers should have minimum 6G:**

Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
for details and troubleshooting.
<pre data-test="exec">
$ docker info | grep "Total Memory" && \
  docker swarm init
</pre>

**Check-out the example repository:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APP=`pwd`/sample-apps/examples/operations/secure-vespa-with-mtls
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
$ $VESPA_SAMPLE_APP/scripts/generate_hosts_xml.sh | tee $VESPA_SAMPLE_APP/hosts.xml
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
$ $VESPA_SAMPLE_APP/scripts/client-curl.sh -s "https://localhost:8443/search/?query=michael" | \
  python3 -m json.tool
</pre>

**Congratulations. You have now deployed and tested a secure Vespa installation.**

**After finished testing the Vespa application, run this to delete the services:**
<pre data-test="after">
$ docker stack rm vespa
</pre>


## Swarm init and network
Example Swarm init:

* Create a swarm on hostname1 using `docker swarm init`
* Join the swarm from other nodes to hostname1 by `docker swarm join --token <token> hostname1-ip:2377`


## Manual network setup notes
Note that the guide above sets up network in [docker-compose.yml](docker-compose.yml).
After joining the swarm, then create a network on hostname1:

    docker network create -d overlay --attachable vespa-net

On all nodes:

    docker run --detach --name nodeXXX --network vespa-net --hostname <hostnameXXX> \
    -e VESPA_CONFIGSERVERS=hostname1,hostname2 \
    --publish 8081:8080 --publish 19071:19071 --publish 19050:19050 --publish 19092:19092 \
    vespaengine/vespa


## Memory settings
For real use, remove settings like

    VESPA_CONFIGSERVER_JVMARGS: -Xms32M -Xmx128M
    VESPA_CONFIGPROXY_JVMARGS: -Xms32M -Xmx32M

from [docker-compose.yml](docker-compose.yml) and

    jvmargs="-Xms32M -Xmx64M"

from [services.xml](services.xml) - these are only added to shrink memory footprint for this guide.
