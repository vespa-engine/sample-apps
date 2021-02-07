#!/bin/bash
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail

get_hosts() {
  docker stack ps --no-trunc --filter "desired-state=running" vespa | grep "vespaengine/vespa" | awk '{print $2"."$1.".vespa_net"}'
}

echo '<?xml version="1.0" encoding="utf-8" ?>' 
echo '<hosts>'
I=0; for n in $(seq 1 3); do echo "  <host name='cfg$n.vespa_net'><alias>config$I</alias></host>"; I=$(($I + 1)); done
I=0; for h in $(get_hosts|grep content|sort); do echo "  <host name='$h'><alias>content$I</alias></host>"; I=$(($I + 1)); done
I=0; for h in $(get_hosts|grep container|sort); do echo "  <host name='$h'><alias>container$I</alias></host>"; I=$(($I + 1)); done
echo '</hosts>'

