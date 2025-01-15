
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - operations

This folder contains sample applications for Vespa deployments in different topologies,
environments, services and platforms.

Vespa users run on a wide range of stacks - we welcome contributions once you got your app running!

Use the AWS multinode quick-start guides in [getting started](https://docs.vespa.ai/en/getting-started.html).

See [vespa.ai/support](https://vespa.ai/support) for how to get help / FAQ / Stack Overflow.
[Admin procedures](https://docs.vespa.ai/en/operations/admin-procedures.html) can be a good read, too.

## CI/CD
Refer to the examples at [CI/CD](CI-CD/)
for how to add continuous integration and continuous delivery/deployment tests to an application.


## Multinode applications

### Multinode
The [Multinode](multinode/) application sets up a three-node application and experiment with node start/stop.
Use status and metrics pages to inspect the system.
Whenever deploying and facing issues, please refer to this application for how to get useful debugging information for support follow-up - i.e. run this application first.

### Multinode High Availability
The [Multinode High Availability](multinode-HA/) application is a great starting point for high-availability multi-node deployment of Vespa.
The example deploys 10 nodes on one base host, and is a good example of a full-fledged multinode application -
use this as a template. This example includes securing the application using mTLS.


### Vespa on Kubernetes 
The [Vespa on Kubernetes (Google Kubernetes Engine)](basic-search-on-gke/)
application demonstrates deploying a simple Vespa application on Kubernetes (Google GKE).

## Monitoring
These are monitoring and metrics-oriented applications.

### Vespa metrics and monitoring
The [Vespa metrics and monitoring](monitoring/album-recommendation-monitoring/)
application demonstrates how to integrate Vespa with **Prometheus and Grafana**.


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

