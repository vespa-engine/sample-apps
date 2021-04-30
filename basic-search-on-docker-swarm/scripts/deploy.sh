#!/bin/bash
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

readonly MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $MYDIR

readonly SID=$(tar -C ../src/main/application -cf - . | gzip | curl -s -L --header "Content-Type: application/x-gzip" --data-binary @- "$(hostname):19071/application/v2/tenant/default/session" | python -c "import sys,json; print(json.load(sys.stdin)['session-id']);")
curl -s -L -X PUT "$(hostname):19071/application/v2/tenant/default/session/$SID/prepared" | python -mjson.tool
curl -s -L -X PUT "$(hostname):19071/application/v2/tenant/default/session/$SID/active" | python -mjson.tool

