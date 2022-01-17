#!/bin/bash
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

readonly MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $MYDIR
while  ! $MYDIR/client-curl.sh -s "https://localhost:8443/search/?query=bad" | python -c "import sys,json; print(json.load(sys.stdin)['root']['coverage']);" &> /dev/null; do
    echo "Search backend not up yet. Sleeping 5 seconds."
    sleep 5
done

$MYDIR/client-curl.sh -s -L -H "Content-Type:application/json" --data-binary @../music-data-1.json "https://localhost:8443/document/v1/music/music/docid/1" | python -m json.tool
$MYDIR/client-curl.sh -s -L -H "Content-Type:application/json" --data-binary @../music-data-2.json "https://localhost:8443/document/v1/music/music/docid/2" | python -m json.tool

