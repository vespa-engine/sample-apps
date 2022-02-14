# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#!/bin/bash

DIR="$1"
echo "[INFO] Downloading models into $DIR"

mkdir -p $DIR

echo "Downloading https://data.vespa.oath.cloud/onnx_models/ms-marco-MiniLM-L-6-v2-quantized.onnx"
curl -L -o $DIR/msmarco_v2.onnx \
https://data.vespa.oath.cloud/onnx_models/ms-marco-MiniLM-L-6-v2-quantized.onnx

echo "Downloading https://data.vespa.oath.cloud/onnx_models/sentence-msmarco-MiniLM-L-6-v3-quantized.onnx"
curl -L -o $DIR/dense_encoder.onnx \
https://data.vespa.oath.cloud/onnx_models/sentence-msmarco-MiniLM-L-6-v3-quantized.onnx

echo "Downloading https://data.vespa.oath.cloud/onnx_models/vespa-colMiniLM-L-6-quantized.onnx"
curl -L -o $DIR/colbert_encoder.onnx \
https://data.vespa.oath.cloud/onnx_models/vespa-colMiniLM-L-6-quantized.onnx
