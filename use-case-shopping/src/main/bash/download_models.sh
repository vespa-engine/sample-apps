# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#!/bin/bash

DIR="$1"
echo "[INFO] Downloading models into $DIR"

mkdir -p $DIR

echo "Downloading encoder model" 
curl -L -o $DIR/minilm-l6-v2.onnx \
https://github.com/vespa-engine/sample-apps/blob/master/simple-semantic-search/model/minilm-l6-v2.onnx?raw=true

echo "Downloading vocab" 
curl -L -o $DIR/bert-base-uncased.txt \
https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/bert-base-uncased.txt
