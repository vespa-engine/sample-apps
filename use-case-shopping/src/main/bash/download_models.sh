# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#!/bin/bash

DIR="$1"
echo "[INFO] Downloading models into $DIR"

mkdir -p $DIR

echo "Downloading encoder model" 
curl -L -o $DIR/e5-small-v2-int8.onnx \
https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/e5-small-v2-int8.onnx

echo "Downloading vocab" 
curl -L -o $DIR/tokenizer.json \
https://data.vespa-cloud.com/sample-apps-data/e5-small-v2-int8/tokenizer.json
