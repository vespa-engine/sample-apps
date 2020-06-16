#!/bin/bash
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail

NB_CONTENTS=${1:-"3"}
NB_CONTAINERS=${2:-"3"}


cat templates/container.yml | sed "s/\NB_CONTAINER/$NB_CONTAINERS/" > ./deployments/container.yml
cat templates/content.yml | sed "s/\NB_CONTENT/$NB_CONTENTS/" > ./deployments/content.yml

CONTENTS="      "
for ((i=0; i<NB_CONTENTS; i++)); do
    line="<node hostalias='content$i' distribution-key='$i' />"
    CONTENTS=$CONTENTS$line
done

CONTAINERS="      "
for ((i=0; i<NB_CONTAINERS; i++)); do
    line="<node hostalias='container$i'/>"
    CONTAINERS=$CONTAINERS"$line"
done

cat templates/services.xml | sed "s|CONTAINER|$CONTAINERS|g" | sed "s|CONTENT|$CONTENTS|g" > ./src/main/application/services.xml

echo '<?xml version="1.0" encoding="utf-8" ?>' > ./src/main/application/hosts.xml
echo '<hosts>' >> ./src/main/application/hosts.xml
echo "  <host name='vespa-0.vespa-internal.default.svc.cluster.local'><alias>admin0</alias><alias>container0</alias><alias>content0</alias></host>" >> ./src/main/application/hosts.xml
for ((i=1; i<NB_CONTENTS; i++)); do echo "  <host name='vespa-content-$i.vespa-internal.default.svc.cluster.local'><alias>content$i</alias></host>" >> ./src/main/application/hosts.xml; done
for ((i=1; i<NB_CONTAINERS; i++)); do echo "  <host name='vespa-container-$i.vespa-internal.default.svc.cluster.local'><alias>container$i</alias></host>" >> ./src/main/application/hosts.xml; done
echo '</hosts>' >> ./src/main/application/hosts.xml
