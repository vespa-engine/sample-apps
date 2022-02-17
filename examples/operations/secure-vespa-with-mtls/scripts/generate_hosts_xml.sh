#!/bin/bash
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail

get_hosts() {
  docker stack ps --no-trunc --filter "desired-state=running" vespa | grep "vespaengine/vespa" | awk '{print $2""}'
}

echo '<?xml version="1.0" encoding="utf-8" ?>'
echo '<hosts>'
for h in $(get_hosts|sort); do
  h="${h%.[0-9]*}"
  h="${h#*_}"
  echo "  <host name='${h}-vespa-net'><alias>${h}</alias></host>"
done
echo '</hosts>'

