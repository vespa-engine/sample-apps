#!/bin/bash
# Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -euo pipefail
set -x

readonly MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $MYDIR

curl -s -L -H "Content-Type:application/json" --data-binary @../music-data-1.json "http://$(hostname):8080/document/v1/music/music/docid/1" | python -m json.tool
curl -s -L -H "Content-Type:application/json" --data-binary @../music-data-2.json "http://$(hostname):8080/document/v1/music/music/docid/2" | python -m json.tool

