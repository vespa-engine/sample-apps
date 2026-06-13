<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Multinode-HA using Helm
This guide uses the multinode-HA configuration and principles and deploys a Vespa application using Kubernetes and Helm.

## Overview
This deployment is designed for high availability and uses the Helm chart consisting of two primary modules:
1. `config` - contains and deploys **vespa-configserver**, which must be running successfully before starting other components.
2. `services` - starts **admin**, **content**, **feed**, and **query** clusters, depending on a successful `configserver` startup.

A key mechanism ensures the correct start order of the modules: `initContainers` in `services` waits for the `configserver` to become ready by repeatedly checking its health. Only after the `configserver` successfully initializes, the `services` module will proceed to start. Here’s the command used in the `initContainers`:

```bash
until curl -f http://vespa-configserver-0.vespa-internal.vespa.svc.cluster.local:19071/state/v1/health; do
  echo "Waiting for Vespa ConfigServer to be ready in namespace $CONFIGSERVER_NAMESPACE...";
  sleep 5;
done
```

---

## Prerequisites
Make sure the following tools are installed and configured:
* [Helm](https://helm.sh/docs/intro/install/)
* A Kubernetes cluster - either local or hosted (e.g., Azure AKS, AWS EKS, etc.)

---

## Installation
Clone the repository:
```bash
git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
cd sample-apps/examples/operations/multinode-HA/helm
```

Prepare your `values.yaml` with the desired configuration. Here's an example:
```yaml
config:
  serverReplicas: 3
services:
  content:
    replicas: 2
    storage: 25Gi
```

Deploy Vespa using Helm:
```bash
helm dependency update helm
helm upgrade --install vespa . -n vespa --create-namespace -f values.yaml
```

This will create the namespace `vespa` and deploy all components of the application.

---

## Deploy the application package

```
kubectl port-forward -n vespa pod/vespa-configserver-0 19071
```

```
(cd conf && zip -r - .) | \
  curl --header Content-Type:application/zip \
  --data-binary @- \
  http://localhost:19071/application/v2/tenant/default/prepareandactivate
```

---

## Module Details

### Config Module
The `config` module contains the **vespa-configserver**, which is essential for Vespa's operation. This module deploys a StatefulSet with `serverReplicas` to ensure high availability.

#### ConfigServer Health Check
The `configserver` health is verified by an HTTP curl to its `/state/v1/health` endpoint. The `services` module will not start until all `configserver` replicas are running and reachable.

Here’s an example of the expected `configserver` health response:
```json
{
    "time" : 1678268549957,
    "status" : {
        "code" : "up"
    },
    "metrics" : {
        "snapshot" : {
            "from" : 1.678268489718E9,
            "to" : 1.678268549718E9
        }
    }
}
```

### Services Module
The `services` module contains the following components:
- **Admin**: Handles Vespa cluster administration.
- **Feed**: Handles document feeding.
- **Query**: Handles document queries.
- **Content**: Stores indexed data.

This module is configured to depend on the `config` module startup. The `initContainers` logic ensures that no pods in the `services` module are started until the `configserver` reaches a stable, healthy state.

Below is the typical `initContainers` logic defined for `services`:
```bash
until curl -f http://vespa-configserver-0.vespa-internal.vespa.svc.cluster.local:19071/state/v1/health; do
  echo "Waiting for Vespa ConfigServer to be ready in namespace $CONFIGSERVER_NAMESPACE...";
  sleep 5;
done
```

---

## Verification
Once the installation completes, you can test the Vespa application by:
1. Feeding a document:
   ```bash
   curl -X POST http://vespa-query-container-0.vespa.svc.cluster.local/document/v1/my-space/my-doc \
       -d '{"id": "id:my-space:my-doc::1", "fields": {"field1": "value1"}}'
   ```

2. Querying documents:
   ```bash
   curl "http://vespa-query-container-0.vespa.svc.cluster.local/search/?query=my-query"
   ```
   
   or
   
   ```
   kubectl -n vespa port-forward svc/vespa-query 8080
   ```
   ```
   curl --data-urlencode 'yql=select * from sources * where true' \
     http://localhost:8080/search/
   ```

---

## Customization and Scaling
Values such as `config.serverReplicas`, `services.content.replicas`, and `services.content.storage` can be adjusted in `values.yaml` to match your requirements for scaling and resource configuration. For example:
```yaml
config:
  serverReplicas: 5
services:
  content:
    replicas: 4
    storage: 50Gi
```

Refer to the official [Vespa documentation](https://docs.vespa.ai/en/) for advanced deployment details and customization options.

---

## Troubleshooting
Check Helm release status to confirm all components deployed successfully:
```bash
helm status vespa -n vespa
```

If pods are stuck, ensure that:
1. The `configserver` is running and reachable.
2. Kubernetes networking allows communication between the pods.

For further troubleshooting details, refer to the [Vespa troubleshooting guide](https://docs.vespa.ai/en/operations.html).