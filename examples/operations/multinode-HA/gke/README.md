<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Multinode-HA using Google Cloud Kubernetes Engine - GKE
This example uses the multinode-HA configuration and principles and deploys GKE.
It is built on [basic-search-on-gke](../../basic-search-on-gke).

This example assumes that you already created a Google project,
you have the [gcloud command line](https://cloud.google.com/sdk/docs/install) and
[kubectl](https://kubernetes.io/docs/tasks/tools/) installed.
If needed, please refer to [GKE quickstart](https://cloud.google.com/kubernetes-engine/docs/deploy-app-cluster).

This guide uses port forwards to access ports in the application -
set up these forwards in separate terminal windows.
The guide does not address setting up load balancers.

Get started:
```
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/examples/operations/multinode-HA/gke
```

Replace with your own project ID and preferred region/zone - example:
```
$ gcloud init
$ gcloud config set project resonant-diode-123456
$ gcloud services enable containerregistry.googleapis.com
$ gcloud config set compute/region europe-west1
$ gcloud config set compute/zone europe-west1-b
```



## Cluster setup
Set up a cluster with 24G RAM:
```
$ gcloud container clusters create vespa \
  --zone europe-west1-b \
  --num-nodes=3 \
  --machine-type=n1-standard-2 \
  --disk-size=20
  
$ gcloud container clusters get-credentials vespa --zone europe-west1-b
```
This is a minimum-configuration to start the multinode-HA application on few resources.



## Config server cluster startup
A Vespa application's nodes are configured using a config server cluster -
everything depends on a successful config server cluster startup, here using three instances:
```
$ kubectl apply \
  -f config/configmap.yml \
  -f config/headless.yml \
  -f config/service.yml \
  -f config/configserver.yml
```

Note that the StatefulSet definition for config servers does not have a `readinessProbe`.
This is important to start all three config servers for zookeeper quorum and subsequent OK status -
[details](https://docs.vespa.ai/en/operations/configuration-server.html#start-sequence).

Pay attention to the config server names, referred in [config/configmap.yml](config/configmap.yml) -
`VESPA_CONFIGSERVERS` is used on all nodes to get configuration.
Make sure all config servers are running well:
```
$ kubectl get pods
NAME                   READY   STATUS    RESTARTS   AGE
vespa-configserver-0   1/1     Running   0          2m25s
vespa-configserver-1   1/1     Running   0          2m13s
vespa-configserver-2   1/1     Running   0          2m4s
```

Check a status page:
```
$ kubectl port-forward pod/vespa-configserver-0 19071:19071
```
```
$ curl http://localhost:19071/state/v1/health
```

Observe status up:
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

If you are not able to see this status page,
do [troubleshooting](https://docs.vespa.ai/en/operations/configuration-server.html#start-sequence).

Note both "configserver,services" are started in [config/configserver.yml](config/configserver.yml)



## Vespa startup
Start the admin node, feed container cluster, query container cluster and content node pods:
```
$ kubectl apply \
  -f config/configmap.yml \
  -f config/headless.yml \
  -f config/service.yml \
  -f config/configserver.yml \
  -f config/admin.yml \
  -f config/feed-container.yml \
  -f config/query-container.yml \
  -f config/content.yml
```

Make sure all pods starts
```
$ kubectl get pods
NAME                      READY   STATUS    RESTARTS   AGE
vespa-admin-0             1/1     Running   0          2m43s
vespa-configserver-0      1/1     Running   0          20m
vespa-configserver-1      1/1     Running   0          20m
vespa-configserver-2      1/1     Running   0          20m
vespa-content-0           1/1     Running   0          2m43s
vespa-content-1           1/1     Running   0          2m36s
vespa-feed-container-0    1/1     Running   0          2m43s
vespa-feed-container-1    1/1     Running   0          2m41s
vespa-query-container-0   1/1     Running   0          2m43s
vespa-query-container-1   1/1     Running   0          2m41s
```

At this point, all pods are started.
The Vespa Application Package is not yet deployed, so none of the Vespa services are running.
Deploy the application package:
```
$ kubectl port-forward pod/vespa-configserver-0 19071:19071
```
```
$ zip -r - . -x "img/*" "scripts/*" "pki/*" "tls/*" README.md .gitignore "config/*" | \
  curl --header Content-Type:application/zip \
  --data-binary @- \
  http://localhost:19071/application/v2/tenant/default/prepareandactivate
```

Expected output:
```json
{
    "log":[],
    "tenant":"default",
    "session-id":"5",
    "url":"http://localhost:19071/application/v2/tenant/default/application/default/environment/prod/region/default/instance/default",
    "message":"Session 5 for tenant 'default' prepared and activated.",
    "configChangeActions":{
        "restart":[],
        "refeed":[],
        "reindex":[]
    }
}
```



## Feed data
Feed data to a feed container:
```
$ kubectl port-forward pod/vespa-feed-container-0 8080:8080
```
```
$ i=0; (for doc in $(ls ../../../../album-recommendation/ext); \
  do \
    curl -H Content-Type:application/json -d @../../../album-recommendation/ext/$doc \
    http://localhost:8080/document/v1/mynamespace/music/docid/$i; \
    i=$(($i + 1)); echo; \
  done)
```



## Run a query
```
$ kubectl port-forward pod/vespa-query-container-0 8081:8080
```
```
$ curl --data-urlencode 'yql=select * from sources * where true' \
  http://localhost:8081/search/
```

**Congratulations! You have now deployed and tested a Vespa application on a multinode cluster.**


## Security
This script is just an example, it exposes the nodes to the internet.
For production purpose you should disable it according to
[vespa security guidelines](https://docs.vespa.ai/en/securing-your-vespa-installation.html)



## Teardown
```
$ gcloud container clusters delete vespa --zone europe-west1-b
```


## Misc
Clean pods for a new deployments:
```
kubectl delete StatefulSet vespa-admin vespa-configserver vespa-content vespa-feed-container vespa-query-container
```
Troubleshoot a pod failing to start up:
```
kubectl describe pod vespa-feed-container-1 - look for "Insufficient memory"
```
Access a port in the application - set up in a separate terminal:
```
kubectl port-forward pod/vespa-query-container-0 8080:8080
```
