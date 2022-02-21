<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)



# Vespa high-availability multi-node template
This sample application is configured using clusters of nodes for HA.
This application structure will scale to applications of 100s of nodes.
Also see the smaller and simpler [multinode](../multinode) sample application.

This is a guide for functional testing, deployed on one host for simplicity.
Refer to the [mTLS guide](/examples/operations/secure-vespa-with-mtls) for a multinode configuration,
set up on multiple hosts using Docker Swarm.
See the [reference documentation](https://docs.vespa.ai/en/reference/metrics.html)
for use of `/state/v1/health` and `/state/v1/metrics` APIs used in this guide.

There are multiple ways of deploying multinode applications, on multiple platforms.
This application is a set of basic Docker containers,
and describes the important elements and integration points.
Use this app as a reference for how to distribute nodes and how to validate the instance:

![Vespa-HA topology](img/vespa-HA.svg)

See [services.xml](src/main/application/services.xml) for the configuration -
in this guide:

* the config cluster is built from 3 config server nodes, node[0,1,2].
  See the `admin` section in services.xml.
* the admin server node that hosts the log server is hosted on node4.
  See the `admin` section in services.xml.
* the stateless java container cluster that hosts the _stateless_ nodes for _feed_ processing on node[4,5].
  See `container` section in services.xml.
* the stateless java container cluster that hosts the _stateless_ nodes for _query_ processing on node[6,7].
  See `container` section in services.xml.
* the content cluster that hosts the _stateful_ content nodes on node[8,9].
  See `content` section in services.xml.

See [Process overview](#process-overview) below for more details,
why the clusters and services are configured in this way.
Also see [troubleshooting](/examples/operations/README.md#troubleshooting).

This guide is tested with Docker using 12G Memory:

<pre data-test="exec">
$ docker info | grep "Total Memory"
</pre>

Note that this guide is configured for minimum memory use for easier testing, adding:

    -e VESPA_CONFIGSERVER_JVMARGS="-Xms32M -Xmx128M" \
    -e VESPA_CONFIGPROXY_JVMARGS="-Xms32M -Xmx32M" \

to `docker run` commands. For real production use cases, do not do this.
Also remove annotated memory-settings in [services.xml](src/main/application/services.xml).

Get the app and create the local network:

<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/examples/operations/multinode-HA
$ docker network create --driver bridge vespanet
</pre>

The goal of this sample application is to create a Docker network with 10 nodes,
exporting ports to be able to observe and validate that the application is fully operational:

![Application overview](img/multinode-HA.svg)

Note that this application uses two container clusters - one for feeding and one for queries.
This allows independent scaling, as reads and writes often have different load characteristics,
e.g. batch writes with high throughput vs. user-driven queries.



## Start the config server cluster
Config servers are normally started first,
as the other Vespa nodes depend on config servers for startup:
<pre data-test="exec">
$ docker run --detach --name node0 --hostname node0.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    -e VESPA_CONFIGSERVER_JVMARGS="-Xms32M -Xmx128M" \
    -e VESPA_CONFIGPROXY_JVMARGS="-Xms32M -Xmx32M" \
    --network vespanet \
    --publish 19071:19071 --publish 19100:19100 --publish 19050:19050 --publish 20092:19092 \
    vespaengine/vespa
$ docker run --detach --name node1 --hostname node1.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    -e VESPA_CONFIGSERVER_JVMARGS="-Xms32M -Xmx128M" \
    -e VESPA_CONFIGPROXY_JVMARGS="-Xms32M -Xmx32M" \
    --network vespanet \
    --publish 19072:19071 --publish 19101:19100 --publish 19051:19050 --publish 20093:19092 \
    vespaengine/vespa
$ docker run --detach --name node2 --hostname node2.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    -e VESPA_CONFIGSERVER_JVMARGS="-Xms32M -Xmx128M" \
    -e VESPA_CONFIGPROXY_JVMARGS="-Xms32M -Xmx32M" \
    --network vespanet \
    --publish 19073:19071 --publish 19102:19100 --publish 19052:19050 --publish 20094:19092 \
    vespaengine/vespa
</pre>

Notes:
* Use fully qualified hostnames.
* VESPA_CONFIGSERVERS lists all nodes using exactly the same names as in
  [hosts.xml](src/main/application/hosts.xml)
* Refer to the
  [Docker start script](https://github.com/vespa-engine/docker-image/blob/master/include/start-container.sh)
  for details.

Vespa separates between starting config servers and service nodes,
see [Vespa start/stop](https://docs.vespa.ai/en/reference/files-processes-and-ports.html#vespa-start-stop).
Normally config servers run both the `config server` _and_ `services`, other nodes run `services` only.
This because `services` has node infrastructure, e.g. log forwarding.

At this point, nothing other than config server cluster runs.
Wait for last config server to start -
checking for 19071 on config server nodes is useful for that
(find the port mappings in the illustration above):
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:19073/ApplicationStatus
</pre>

    HTTP/1.1 200 OK
    Date: Tue, 02 Nov 2021 10:20:10 GMT
    Content-Type: application/json
    Content-Length: 13109



## Deploy the Vespa application configuration
<!-- ToDo: vespa-cli -->
<pre data-test="exec" data-test-assert-contains="prepared and activated.">
$ (cd src/main/application && zip -r - .) | \
  curl --header Content-Type:application/zip --data-binary @- \
  localhost:19071/application/v2/tenant/default/prepareandactivate
</pre>

As the Docker start script will start both the config server and services,
we expect to find responses on 19100 (`slobrok`) and 19050 (`cluster-controller`).

**Important note:** Vespa has a feature to not enable `services` before 50% of the nodes are up,
see [startup sequence](https://docs.vespa.ai/en/config-sentinel.html#cluster-startup).
Meaning, here we have started only 3/10, so `slobrok` and `cluster-controller` are not started yet -
find log messages like

    $ docker exec -it node0 sh -c "opt/vespa/bin/vespa-logfmt | grep config-sentinel | tail -5"

      WARNING : config-sentinel  sentinel.sentinel.connectivity Only 3 of 10 nodes are up and OK, 30.0% (min is 50%)
      WARNING : config-sentinel  sentinel.sentinel.env Bad network connectivity (try 71)



## Start admin server
This is essentially the log server:
<pre data-test="exec">
$ docker run --detach --name node3 --hostname node3.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    --network vespanet \
    --publish 20095:19092 \
    vespaengine/vespa services
</pre>

Notes:
* See  _services_ argument to start script - a config server is not started on this node.
* The log server can be disk intensive, as vespa.log is rotated and forwarded here.
  It is normally a good idea to run this on a separate node, like here, for this reason -
  a full disk will not hit the other nodes then. 



## Start the feed container cluster
The feed container cluster has the feed endpoint, normally on 8080.
<pre data-test="exec">
$ docker run --detach --name node4 --hostname node4.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    --network vespanet \
    --publish 8080:8080 --publish 20096:19092 \
    vespaengine/vespa services
$ docker run --detach --name node5 --hostname node5.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    --network vespanet \
    --publish 8081:8080 --publish 20097:19092 \
    vespaengine/vespa services
</pre>

Check for OK startup - this can take a minute or so:
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8081/ApplicationStatus
</pre>

    HTTP/1.1 200 OK
    Date: Wed, 19 Jan 2022 08:29:23 GMT
    Content-Type: application/json
    Content-Length: 5213

As this is also the cluster where custom components for document processing are loaded,
inspecting the `/ApplicationStatus` endpoint is useful:

    $ curl http://localhost:8080/ApplicationStatus

As these are nodes 5 and 6 of 10, 60% of services is started:

    $ docker exec -it node0 sh -c "opt/vespa/bin/vespa-logfmt | grep config-sentinel | tail -5"

      WARNING : config-sentinel  sentinel.vespalib.net.async_resolver could not resolve host name: 'node8.vespanet'
      WARNING : config-sentinel  sentinel.vespalib.net.async_resolver could not resolve host name: 'node9.vespanet'
      INFO    : config-sentinel  sentinel.sentinel.connectivity Connectivity check details: node4.vespanet -> OK: both ways connectivity verified
      INFO    : config-sentinel  sentinel.sentinel.connectivity Connectivity check details: node5.vespanet -> OK: both ways connectivity verified
      INFO    : config-sentinel  sentinel.sentinel.connectivity Enough connectivity checks OK, proceeding with service startup

We hence expect the `metric proxy` (runs on all service nodes) to be up - and others:

    $ curl http://localhost:20095/state/v1/   # metrics proxy on node3
    $ curl http://localhost:19100/state/v1/   # slobok on node0
    $ curl http://localhost:19051/state/v1/   # cluster-controller on node1

In short, checking `/state/v1/` for services is useful to validate that services are up.
At this point it is useful to inspect the application using
[vespa-model-inspect](https://docs.vespa.ai/en/reference/vespa-cmdline-tools.html#vespa-model-inspect):
```sh
$ docker exec -it node0 sh -c "/opt/vespa/bin/vespa-model-inspect hosts"
node0.vespanet
node1.vespanet
node2.vespanet
node3.vespanet
node4.vespanet
node5.vespanet
node6.vespanet
node7.vespanet
node8.vespanet
node9.vespanet

$ docker exec -it node0 sh -c "/opt/vespa/bin/vespa-model-inspect services"
config-sentinel
configproxy
configserver
container
container-clustercontroller
distributor
logd
logserver
metricsproxy-container
qrserver
searchnode
slobrok
storagenode
transactionlogserver

$ docker exec -it node0 sh -c "/opt/vespa/bin/vespa-model-inspect service container"
container @ node4.vespanet : 
default/container.0
    tcp/node4.vespanet:8080 (STATE EXTERNAL QUERY HTTP)
    tcp/node4.vespanet:19100 (EXTERNAL HTTP)
    tcp/node4.vespanet:19101 (MESSAGING RPC)
    tcp/node4.vespanet:19102 (ADMIN RPC)
container @ node5.vespanet : 
default/container.1
    tcp/node5.vespanet:8080 (STATE EXTERNAL QUERY HTTP)
    tcp/node5.vespanet:19100 (EXTERNAL HTTP)
    tcp/node5.vespanet:19101 (MESSAGING RPC)
    tcp/node5.vespanet:19102 (ADMIN RPC)
```

This tool only uses a config server, it does not check other nodes -
it displays a view of the layout of the application - what runs where.

**Important takeaway:** Vespa port allocations are dynamic.
This enables Vespa to run a set of processes as configured in services.xml on each node.
Use `vespa-model-inspect` to find where processes run and which ports to use -
this tool can be used on any Vespa node.



## Start the query container cluster
The query container cluster has the query endpoint, normally on 8080.
<pre data-test="exec">
$ docker run --detach --name node6 --hostname node6.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    --network vespanet \
    --publish 8082:8080 --publish 20098:19092 \
    vespaengine/vespa services
$ docker run --detach --name node7 --hostname node7.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    --network vespanet \
    --publish 8083:8080 --publish 20099:19092 \
    vespaengine/vespa services
</pre>

Again, wait for OK startup:
<pre data-test="exec" data-test-wait-for="200 OK">
$ curl -s --head http://localhost:8083/ApplicationStatus
</pre>



## Start the content cluster:
<pre data-test="exec">
$ docker run --detach --name node8 --hostname node8.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    --network vespanet \
    --publish 19107:19107 --publish 20100:19092 \
    vespaengine/vespa services
$ docker run --detach --name node9 --hostname node9.vespanet \
    -e VESPA_CONFIGSERVERS=node0.vespanet,node1.vespanet,node2.vespanet \
    --network vespanet \
    --publish 19108:19107 --publish 20101:19092 \
    vespaengine/vespa services
</pre>

Check for content node startup using `/state/v1/health`:
<pre data-test="exec" data-test-wait-for='"code":"up"'>
$ curl http://localhost:19108/state/v1/health
</pre>
```json
{
    "status": {
        "code": "up"
    }
}
```

Inspect the content node _process_ metrics using `/state/v1/metrics`:
<pre data-test="exec" data-test-wait-for='"metrics":  '>
$ curl http://localhost:19108/state/v1/metrics
</pre>
```json
{
    "status": {
        "code": "up"
    },
    "metrics": {
        "snapshot": {
            "from": 1643285217,
            "to": 1643285277
        },
        "values": [
            {
                "name": "metricmanager.periodichooklatency",
                "description": "Time in ms used to update a single periodic hook",
                "values": {
                    "average": 0,
                    "sum": 0,
                    "count": 255,
                    "rate": 4.25,
                    "min": 0,
                    "max": 0,
```

Dump the content node _node_ metrics (i.e. _all_ processes on the node) using `/metrics/v1/values`:
<pre data-test="exec" data-test-wait-for='"name":"vespa.searchnode"'>
$ curl http://localhost:20101/metrics/v1/values
</pre>
```json
{
    "services": [
        {
            "name": "vespa.searchnode",
            "timestamp": 1643284794,
            "status": {
                "code": "up",
                "description": "Data collected successfully"
        },
        "metrics": [
```

Dump the content node _node_ metrics, in Prometheus format:
<pre data-test="exec" data-test-wait-for='"name":"vespa.searchnode"'>
$ curl http://localhost:20101/prometheus/v1/values
</pre>
    # HELP memory_virt
    # TYPE memory_virt untyped
    memory_virt{metrictype="system",instance="distributor",vespaVersion="7.531.17",vespa_service="vespa_distributor",} 3.39165184E8 1643286737000
    memory_virt{metrictype="system",instance="logd",vespaVersion="7.531.17",vespa_service="vespa_logd",} 1.29429504E8 1643286737000

Test metrics from all nodes using `/metrics/v2/values`:
<pre data-test="exec" data-test-wait-for='"services":'>
curl http://localhost:8083/metrics/v2/values
</pre>
```json
{
    "nodes": [
        {
            "hostname": "node5.vespanet",
            "role": "hosts/node5.vespanet",
            "services": [
                {
                    "name": "vespa.container",
                    "timestamp": 1643289142,
                    "status": {
```

## Process overview
Notes:
* [Config sentinel](https://docs.vespa.ai/en/config-sentinel.html)
  is useful to understand processes running,
  and also the [startup sequence](https://docs.vespa.ai/en/config-sentinel.html#cluster-startup).
  Note that in the startup sequence, order not strictly needed as in this sample app.
* [Config servers](https://docs.vespa.ai/en/cloudconfig/configuration-server.html) are normally started first,
  then application deployment - make sure to get this right before troubleshooting other services.
* See [slobrok](https://docs.vespa.ai/en/slobrok.html) for the Vespa naming service
* The [cluster controller](https://docs.vespa.ai/en/content/content-nodes.html#cluster-controller) cluster
  manages the system state, and is useful in debugging cluster failures.
* The [metrics proxy](https://docs.vespa.ai/en/reference/metrics.html) is used to aggregate metrics 
  from all processes on a node, serving on _http://node:19092/metrics/v1/values_



## Test endpoints
Feed 5 documents, using the document-API endpoint in the _feed_ container cluster on 8080/8081:
<pre data-test="exec">
$ i=0; (for doc in $(ls ../../../album-recommendation/src/test/resources); \
  do \
    curl -H Content-Type:application/json -d @../../../album-recommendation/src/test/resources/$doc \
    http://localhost:8080/document/v1/mynamespace/music/docid/$i; \
    i=$(($i + 1)); echo; \
  done)
</pre>

List IDs of all documents (this can be run on any node in the cluster):
<pre data-test="exec" data-test-wait-for="id:mynamespace:music::4">
$ docker exec node0 bash -c "/opt/vespa/bin/vespa-visit -i"
</pre>

Run a query, using the query-API endpoint in the _query_ container cluster on 8082/8083::
<pre data-test="exec" data-test-wait-for='"totalCount":5'>
$ curl --data-urlencode 'yql=select * from sources * where sddocname contains "music"' \
  http://localhost:8082/search/
</pre>



## Clean up after testing
<pre data-test="after">
$ docker rm -f node0 node1 node2 node3 node4 node5 node6 node7 node8 node9
$ docker network rm vespanet
</pre>



## Integrations
[Vispana](https://github.com/vispana/vispana) is a Vespa.ai web client tool
designed to quickly understand the status of a cluster.
To use Vispana with this example app, add it to the same Docker network:

    docker run -p 4000:4000 --network vespanet vispana/vispana

Use _http://node0.vespanet:19071_ as the config server endpoint:

    http://localhost:4000/content?config_host=http://node0.vespanet:19071/

![Vispana screenshot](img/vispana.png)



## Appendix

### Deploy output
Normal deploy output in this guide, as the service nodes are not started yet:
```json
{
  "log": [
    {
      "time": 1642578267697,
      "level": "WARNING",
      "message": "Unable to lookup IP address of host: node4.vespanet",
      "applicationPackage": true
    },
    {
      "time": 1642578268248,
      "level": "WARNING",
      "message": "Unable to lookup IP address of host: node5.vespanet",
      "applicationPackage": true
    },
    {
      "time": 1642578268701,
      "level": "WARNING",
      "message": "Unable to lookup IP address of host: node6.vespanet",
      "applicationPackage": true
    },
    {
      "time": 1642578269060,
      "level": "WARNING",
      "message": "Unable to lookup IP address of host: node7.vespanet",
      "applicationPackage": true
    },
    {
      "time": 1642578269444,
      "level": "WARNING",
      "message": "Unable to lookup IP address of host: node3.vespanet",
      "applicationPackage": true
    },
    {
      "time": 1642578269870,
      "level": "WARNING",
      "message": "Unable to lookup IP address of host: node8.vespanet",
      "applicationPackage": true
    },
    {
      "time": 1642578270221,
      "level": "WARNING",
      "message": "Unable to lookup IP address of host: node9.vespanet",
      "applicationPackage": true
    }
  ],
  "tenant": "default",
  "url": "http://localhost:19071/application/v2/tenant/default/application/default/environment/prod/region/default/instance/default",
  "message": "Session 2 for tenant 'default' prepared and activated.",
  "configChangeActions": {
    "restart": [],
    "refeed": [],
    "reindex": []
  }
}
```


### Ports
Ports mapped in this guide:
```sh
$ netstat -an | egrep '1907[1,2,3]|1905[0,1,2]|2009[2,3,4,5,6,7,8,9]|2010[0,1]|1910[0,1,2]|808[0,1,2,3]|1910[7,8]' | sort
tcp46      0      0  *.19050                *.*                    LISTEN     
tcp46      0      0  *.19051                *.*                    LISTEN     
tcp46      0      0  *.19052                *.*                    LISTEN     
tcp46      0      0  *.19071                *.*                    LISTEN     
tcp46      0      0  *.19072                *.*                    LISTEN     
tcp46      0      0  *.19073                *.*                    LISTEN     
tcp46      0      0  *.19100                *.*                    LISTEN     
tcp46      0      0  *.19101                *.*                    LISTEN     
tcp46      0      0  *.19102                *.*                    LISTEN     
tcp46      0      0  *.19107                *.*                    LISTEN     
tcp46      0      0  *.19108                *.*                    LISTEN     
tcp46      0      0  *.20092                *.*                    LISTEN     
tcp46      0      0  *.20093                *.*                    LISTEN     
tcp46      0      0  *.20094                *.*                    LISTEN     
tcp46      0      0  *.20095                *.*                    LISTEN     
tcp46      0      0  *.20096                *.*                    LISTEN     
tcp46      0      0  *.20097                *.*                    LISTEN     
tcp46      0      0  *.20098                *.*                    LISTEN     
tcp46      0      0  *.20099                *.*                    LISTEN     
tcp46      0      0  *.20100                *.*                    LISTEN     
tcp46      0      0  *.20101                *.*                    LISTEN     
tcp46      0      0  *.8080                 *.*                    LISTEN     
tcp46      0      0  *.8081                 *.*                    LISTEN     
tcp46      0      0  *.8082                 *.*                    LISTEN     
tcp46      0      0  *.8083                 *.*                    LISTEN     
```
