# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#!/bin/bash

DIR="$1"
mkdir -p $DIR
FILE=$DIR/text_transformer.onnx

if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    echo "Downloading model https://data.vespa-cloud.com/sample-apps-data/clip_text_transformer.onnx" 
    curl -L -o $DIR/text_transformer.onnx \
      https://data.vespa-cloud.com/sample-apps-data/clip_text_transformer.onnx
fi

