#!/usr/bin/env bash

# Copyright 2020 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

set -e

docker run --rm -e ALBUM_RECOMMENDATION_JAVA_DEPLOY_KEY -v $(pwd):/source -w /source --entrypoint /source/travis/deploy-album-recommendation-java.sh vespaengine/vespa-pipeline
