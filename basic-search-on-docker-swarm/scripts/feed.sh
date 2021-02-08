#!/bin/bash
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

readonly MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $MYDIR

while  ! curl -s "http://$(hostname):8080/search/?query=bad" | python -c "import sys,json; print json.load(sys.stdin)['root']['coverage'];" &> /dev/null; do
    echo "Search backend not up yet. Sleeping 5 seconds."
    sleep 5
done    

curl -s -L -H "Content-Type:application/json" --data-binary @../music-data-1.json "http://$(hostname):8080/document/v1/music/music/docid/1" | python -m json.tool
curl -s -L -H "Content-Type:application/json" --data-binary @../music-data-2.json "http://$(hostname):8080/document/v1/music/music/docid/2" | python -m json.tool

