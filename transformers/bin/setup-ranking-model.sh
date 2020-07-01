#! /bin/bash

DIR="src/main/application/"
MODEL_NAME=$1

echo "Setting up ranking model..."
./src/python/setup-model.py $DIR $MODEL_NAME
echo "Done setting up model..."

