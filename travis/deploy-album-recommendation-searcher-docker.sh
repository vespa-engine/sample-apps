#!/usr/bin/env bash

# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

set -e

docker run --rm -e ALBUM_RECOMMENDATION_JAVA_DEPLOY_KEY -v $(pwd):/source -w /source --entrypoint /source/travis/deploy-album-recommendation-searcher.sh vespaengine/vespa-pipeline
