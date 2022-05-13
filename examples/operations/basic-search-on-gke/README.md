<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa basic search example on GKE

Please refer to
[Vespa quick start using Docker](https://docs.vespa.ai/en/vespa-quick-start.html)
for more information on the basic single container example.
Also see [Vespa quick start using Kubernetes](https://docs.vespa.ai/en/vespa-quick-start-kubernetes.html).

Containers running Vespa should have minimum 4 GB memory available.
Refer to [Docker memory](https://docs.vespa.ai/en/operations/docker-containers.html#memory)
for details and troubleshooting.

This example assumes that you already created a Google project, you have the gcloud command line and kubectl installed.
If needed, please refer to [GKE quickstart](https://cloud.google.com/kubernetes-engine/docs/quickstart).

The example below shows the steps to pop a GKE cluster and deploy a multinode setup on GKE.

**Note:** To extend this to multiple config servers, set `publishNotReadyAddress: true` on the service,
see [troubleshooting](/examples/operations/README.md#troubleshooting).


### Executable example
**Check-out the example repository:**
<pre data-test="exec">
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ VESPA_SAMPLE_APP=`pwd`/sample-apps/examples/operations/basic-search-on-gke
</pre>

**Create a GKE cluster :**
You can give arguments to this script to change, cluster name, number of nodes, nodes type, and region.
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/create_cluster.sh
</pre>

**Boostrap config files:**
You can give arguments to this script to change number of containers and contents.
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/boostrap.sh
</pre>

**Deploy the application:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/deploy.sh
</pre>

**Feed data to the application:**
<pre data-test="exec">
$ $VESPA_SAMPLE_APP/scripts/feed.sh
</pre>
**Do a search:**
<pre data-test="exec">
$ curl -s "http://$(kubectl get service/vespa -o jsonpath='{.status.loadBalancer.ingress[*].ip}'):$(kubectl get service/vespa -o jsonpath='{.spec.ports[?(@.name=="container")].port}')/search/?query=michael" | \
  python3 -m json.tool
</pre>

**Security notice**
This script is just an example, it'll expose the master node to internet.
For production purpose you should disable it according to
[vespa security guidelines](https://docs.vespa.ai/en/securing-your-vespa-installation.html)

**Congratulations! You have now deployed and tested a Vespa application on a multinode cluster.**

**After you have finished testing the Vespa application, execute the following to delete the cluster:**
Replace CLUSTER_NAME and ZONE with your own values. By default `CLUSTER_NAME=vespa` and `ZONE=europe-west1-b`
<pre data-test="after">
$ gcloud container clusters delete CLUSTER_NAME --zone ZONE
</pre>
