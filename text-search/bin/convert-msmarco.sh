#! /bin/bash

DIR="msmarco"
DOCTYPE="msmarco"
NUM_QUERIES=1000
NUM_DOCS=100000

echo "Extracting documents, queries and relevance judgments. This can take some time..."
./src/python/extract-msmarco.py
echo "Done extracting dataset. Download dir $DIR/download can be removed."

echo "Sampling from corpus..."
./src/python/sample-queries-and-documents.py $DIR $NUM_QUERIES $NUM_DOCS
echo "Done sampling from corpus."

echo "Converting to Vespa format..."
./src/python/convert-to-vespa-format.py $DIR $DOCTYPE id,url,title,body
echo "Done converting to Vespa format."


