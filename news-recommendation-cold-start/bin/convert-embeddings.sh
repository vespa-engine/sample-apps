#! /bin/bash

DIR="mind"

echo "Converting news and user embeddings to Vespa format..."
./src/python/convert_embeddings_to_vespa_format.py $DIR
echo "Done converting to Vespa format."


