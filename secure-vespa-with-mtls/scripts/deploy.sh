#!/bin/bash
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

readonly MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $MYDIR

tar -C ../src/main/application -cf - . | gzip | $MYDIR/vespa-curl.sh -s -L --header "Content-Type: application/x-gzip" --data-binary @- "https://localhost:19071/application/v2/tenant/default/prepareandactivate" | python -mjson.tool
