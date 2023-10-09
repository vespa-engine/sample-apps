#!/bin/bash
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

HOSTNAME=$(kubectl get service/vespa -o jsonpath='{.status.loadBalancer.ingress[*].ip}')
PORT=$(kubectl get service/vespa -o jsonpath='{.spec.ports[?(@.name=="container")].port}')

curl -s -L -H "Content-Type:application/json" \
  --data-binary @ext/music-data-1.json \
  "http://$HOSTNAME:$PORT/document/v1/mynamespace/music/docid/1" | python3 -m json.tool
curl -s -L -H "Content-Type:application/json" \
  --data-binary @ext/music-data-2.json \
  "http://$HOSTNAME:$PORT/document/v1/mynamespace/music/docid/2" | python3 -m json.tool
