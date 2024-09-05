#!/usr/bin/env sh

DIR="$1"
echo "[INFO] Downloading model into $DIR"

mkdir -p $DIR

echo "Downloading https://data.vespa-cloud.com/onnx_models/clip_transformer.onnx"
curl -L -o $DIR/transformer.onnx \
https://data.vespa-cloud.com/onnx_models/clip_transformer.onnx
