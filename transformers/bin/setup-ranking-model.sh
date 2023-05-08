#! /bin/bash

FILE="application/files/ranking_model.onnx"

echo "Setting up ranking model..."
./bin/setup-model.py $FILE
echo "Done setting up model..."
