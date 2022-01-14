<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - operations

This folder contains sample applications for Vespa deployments in different topologies,
environments, services and platforms.

Vespa users run on a wide range of stacks - we welcome contributions once you got your app running!

See [vespa.ai/support](https://vespa.ai/support) for how to get help / FAQ / Stack Overflow.
[Admin procedures](https://docs.vespa.ai/en/operations/admin-procedures.html) can be a good read, too



## Troubleshooting


### Startup problems in multinode Kubernetes cluster
Developers have reported problems running a multinode Vespa cluster on Kubernetes.
Root cause is often the three Vespa Config Servers that uses a shared ZooKeeper cluster,
see [multinode](multinode) for details.
The Zookeeper cluster must be up before all Config Servers are up on http://host:19071/ApplicationStatus.
If 19071 is used in the readinessProbe, the Kubernetes service does not register the host, and there is a catch-22.
Removing the readinessProbe on 19071 can make the Config Server cluster start.
Related: [basic-search-on-gke](../basic-search-on-gke) and 
[vespa-quick-start-kubernetes](https://docs.vespa.ai/en/vespa-quick-start-kubernetes.html)



### Starting Vespa using Docker on M1 fails
Also reported in [#20516](https://github.com/vespa-engine/vespa/issues/20516).
Users on M1 MacBook Pros might see this when starting the Docker container:
```
WARNING: The requested image’s platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)
and no specific platform was requested
```
There is currently no solution to this problem.
Vespa is not yet tested on M1 arch64, pending resources to do so.
