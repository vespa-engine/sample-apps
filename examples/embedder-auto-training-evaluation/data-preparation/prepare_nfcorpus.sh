#!/usr/bin/env bash

mkdir -p datasets/nfcorpus && cd datasets/nfcorpus
ir_datasets export beir/nfcorpus docs  > docs
ir_datasets export beir/nfcorpus docs --format jsonl \
  | python3 ../../data-preparation/prepend-passage.py \
  | python3 ../../data-preparation/add-title-if-missing.py \
  > docs.jsonl
gzip < docs.jsonl > passages.jsonl.gz
< docs.jsonl python3 ../../data-preparation/convert-for-feeding.py > feed.jsonl

ir_datasets export beir/nfcorpus/train qrels > train-qrels
ir_datasets export beir/nfcorpus/train queries > train-queries
ir_datasets export beir/nfcorpus/dev qrels > dev-qrels
ir_datasets export beir/nfcorpus/dev queries > dev-queries
ir_datasets export beir/nfcorpus/test qrels > test-qrels
ir_datasets export beir/nfcorpus/test queries > test-queries
