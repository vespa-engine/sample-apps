#!/bin/bash
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail

NB_CONTENTS=${1:-"3"}
NB_CONTAINERS=${2:-"3"}

NB_CONTENTS=$((NB_CONTENTS-1))
NB_CONTAINERS=$((NB_CONTAINERS-1))

cat templates/container.yml | sed "s/\NB_CONTAINER/$NB_CONTAINERS/" > ./deployments/container.yml
cat templates/content.yml | sed "s/\NB_CONTENT/$NB_CONTENTS/" > ./deployments/content.yml

CONTENTS="      <node hostalias='content' distribution-key='0' />"
for ((i=0; i<NB_CONTENTS; i++)); do
    line="<node hostalias='content$i' distribution-key='$((i+1))' />"
    CONTENTS=$CONTENTS$line
done

CONTAINERS="      <node hostalias='container'/>"
for ((i=0; i<NB_CONTAINERS; i++)); do
    line="<node hostalias='container$i'/>"
    CONTAINERS=$CONTAINERS"$line"
done

cat templates/services.xml | sed "s|CONTAINER|$CONTAINERS|g" | sed "s|CONTENT|$CONTENTS|g" > ./services.xml

echo '<?xml version="1.0" encoding="utf-8" ?>' > ./hosts.xml
echo '<hosts>' >> ./hosts.xml
echo "  <host name='vespa-0.vespa-internal.default.svc.cluster.local'><alias>admin0</alias><alias>container</alias><alias>content</alias></host>" >> ./hosts.xml
for ((i=0; i<NB_CONTENTS; i++)); do echo "  <host name='vespa-content-$i.vespa-internal.default.svc.cluster.local'><alias>content$i</alias></host>" >> ./hosts.xml; done
for ((i=0; i<NB_CONTAINERS; i++)); do echo "  <host name='vespa-container-$i.vespa-internal.default.svc.cluster.local'><alias>container$i</alias></host>" >> ./hosts.xml; done
echo '</hosts>' >> ./hosts.xml
