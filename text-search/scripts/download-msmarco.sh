#! /bin/bash

DIR="ext/download"

mkdir -p $DIR

curl -L -o $DIR/msmarco-doctrain-queries.tsv.gz https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
curl -L -o $DIR/msmarco-doctrain-qrels.tsv.gz https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
curl -L -o $DIR/msmarco-docs-lookup.tsv.gz https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz
curl -L -o $DIR/msmarco-docs.tsv.gz https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz

gunzip $DIR/msmarco-docs.tsv.gz
