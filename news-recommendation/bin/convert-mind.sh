#! /bin/bash

DIR="mind"

echo "Converting train and dev data to Vespa format..."
./src/python/convert_to_vespa_format.py $DIR
echo "Done converting to Vespa format."


