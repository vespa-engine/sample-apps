<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - operations

This folder contains sample applications for Vespa deployments in different topologies,
environments, services and platforms.

Vespa users run on a wide range of stacks - we welcome contributions once you got your app running!

Use the AWS multinode quick-start guides in [getting started](https://docs.vespa.ai/en/getting-started.html).

See [vespa.ai/support](https://vespa.ai/support) for how to get help / FAQ / Stack Overflow.
[Admin procedures](https://docs.vespa.ai/en/operations/admin-procedures.html) can be a good read, too.



## Multinode applications


### [Multinode](multinode)
Set up a three-node application and experiment with node start/stop.
Use status and metrics pages to inspect the system.
Whenever deploying and facing issues, please refer to this application for how to get useful debugging information
for support follow-up - i.e. run this application first.


### [Multinode High Availability](multinode-HA)
Use this application as a starting point for high-availability multi-node applications.
The example deploys 10 nodes on one base host, and is a good example of a full-fledged multinode application -
use this as a template.
This example includes securing the application using mTLS.


### [Vespa on Kubernetes (Google Kubernetes Engine)](basic-search-on-gke)
This sample application demonstrates deploying a simple Vespa application on Kubernetes.



## Monitoring

### [Vespa metrics and monitoring](album-recommendation-monitoring)
This sample app demonstrates how to integrate Vespa with **Prometheus and Grafana**.



## Troubleshooting


### Startup problems in multinode Kubernetes cluster
Developers have reported problems running a multinode Vespa cluster on Kubernetes.
Root cause is often the three Vespa Config Servers that uses a shared ZooKeeper cluster,
see [multinode](multinode) for details.
The Zookeeper cluster must be up before all Config Servers are up on http://host:19071/ApplicationStatus.
If 19071 is used in the readinessProbe, the Kubernetes service does not register the host, and there is a catch-22.
Removing the readinessProbe on 19071 can make the Config Server cluster start /
set `publishNotReadyAddress: true` - also see [basic-search-on-gke](basic-search-on-gke) and 
[vespa-quick-start-kubernetes](https://docs.vespa.ai/en/vespa-quick-start-kubernetes.html).



### Starting Vespa using Docker on M1 fails
Also reported in [#20516](https://github.com/vespa-engine/vespa/issues/20516).
Users on M1 MacBook Pros might see this when starting the Docker container:
```
WARNING: The requested image’s platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)
and no specific platform was requested
```
Vespa is not yet tested on M1 arch64, pending resources to do so. There is a [Preview of Vespa on ARM64](https://blog.vespa.ai/preview-of-vespa-on-arm64/) available.
