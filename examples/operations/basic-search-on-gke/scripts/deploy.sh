#!/bin/bash
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

kubectl apply -f deployments/configmap.yml -f deployments/master.yml -f deployments/container.yml -f deployments/content.yml -f deployments/headless.yml -f deployments/service.yml
while [[ $(kubectl get pods -l app=vespa -o 'jsonpath={..status.conditions[?(@.type=="Ready")].status}' | sort -u) != "True" ]]; do echo "waiting for pod" && sleep 10; done
kubectl cp hosts.xml vespa-0:/workspace
kubectl cp services.xml vespa-0:/workspace
kubectl cp schemas vespa-0:/workspace
kubectl exec vespa-0 -- bash -c '/opt/vespa/bin/vespa-deploy prepare /workspace && /opt/vespa/bin/vespa-deploy activate'
kubectl exec vespa-0 -- bash -c 'while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' http://localhost:8080/ApplicationStatus)" != "200" ]]; do sleep 10; done'
