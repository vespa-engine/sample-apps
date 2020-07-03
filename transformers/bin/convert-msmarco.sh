#! /bin/bash

DIR="msmarco"
DOCTYPE="msmarco"
NUM_QUERIES=10
NUM_DOCS=1000
MODEL_NAME=$1

echo "Extracting documents, queries and relevance judgments. This can take some time..."
./src/python/extract-msmarco.py
echo "Done extracting dataset. Download dir $DIR/download can be removed."

echo "Sampling from corpus..."
./src/python/sample-queries-and-documents.py $DIR $NUM_QUERIES $NUM_DOCS
echo "Done sampling from corpus."

echo "Converting to Vespa format (including tokenizing)..."
./src/python/convert-to-vespa-format.py $DIR $DOCTYPE id,url,title,body $MODEL_NAME $SEQUENCE_LENGTH
echo "Done converting to Vespa format."


