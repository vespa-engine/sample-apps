#! /bin/bash

DIR="msmarco/download"

mkdir -p $DIR

wget -P $DIR -nd https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
wget -P $DIR -nd https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
wget -P $DIR -nd https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz
wget -P $DIR -nd https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz

gunzip $DIR/msmarco-docs.tsv.gz

