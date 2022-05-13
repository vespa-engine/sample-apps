#!/bin/bash
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

readonly MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $MYDIR

(cd .. && zip -r - . -x "tls/*" "scripts/*" "pki/*" "ext/*" README.md .gitignore docker-compose.yml .DS_Store) | \
  $MYDIR/vespa-curl.sh -s -L --header "Content-Type:application/zip" --data-binary @- \
  "https://localhost:19071/application/v2/tenant/default/prepareandactivate" | python3 -m json.tool
