#!/bin/bash
# Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail

readonly VESPA_SERVICES=$(docker stack services vespa|grep "replicated.*[0-9]/[0-9]"|wc -l)
if [ $VESPA_SERVICES -eq 0 ]; then
  echo "No Vespa stack found. Exiting with status 1."
  exit 1
fi

while true; do
  VESPA_SERVICES_NOT_READY=$(docker stack services vespa |grep "replicated.*[0-9]/[0-9]"|awk '{split($4,a,"/"); if (a[1] != a[2]) print "NOT_READY_YET"; }' |wc -l)
  if [ $VESPA_SERVICES_NOT_READY -ne 0 ]; then
    echo "$VESPA_SERVICES_NOT_READY Vespa Docker services not ready. It might take some time to download images on new nodes. Sleeping for one second."
    sleep 1
  else
    break
  fi
done

echo "All Vespa Docker services running."
