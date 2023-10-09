#!/bin/bash
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

kubectl apply \
  -f deployments/configmap.yml \
  -f deployments/master.yml \
  -f deployments/container.yml \
  -f deployments/content.yml \
  -f deployments/headless.yml \
  -f deployments/service.yml

while [[ $(kubectl get pods -l app=vespa -o 'jsonpath={..status.conditions[?(@.type=="Ready")].status}' | sort -u) != "True" ]]
do
  echo "waiting for pods..." && sleep 10
done

CFG_SRV_HOST="$(kubectl get service/vespa -o jsonpath='{.status.loadBalancer.ingress[*].ip}')"
CFG_SRV_PORT="$(kubectl get service/vespa -o jsonpath='{.spec.ports[?(@.name=="config")].port}')"
CFG_SRV_ENDPOINT="$CFG_SRV_HOST:$CFG_SRV_PORT"

zip -r - . \
  -x .gitignore "deployments/*" "ext/*" "scripts/*" "templates/*" README.md "config/*" | \
  curl --header Content-Type:application/zip \
  --data-binary @- \
  "http://$CFG_SRV_ENDPOINT/application/v2/tenant/default/prepareandactivate"

kubectl exec vespa-0 -- bash -c \
  'while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' http://localhost:8080/ApplicationStatus)" != "200" ]]
  do
    sleep 10
  done'
