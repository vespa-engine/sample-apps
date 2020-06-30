#! /bin/bash

DIR="src/main/application/"
SEQUENCE_LENGTH=128
MODEL_NAME="nboost/pt-tinybert-msmarco"

echo "Setting up ranking model..."
./src/python/setup-model.py $DIR $MODEL_NAME $SEQUENCE_LENGTH
echo "Done setting up model..."

