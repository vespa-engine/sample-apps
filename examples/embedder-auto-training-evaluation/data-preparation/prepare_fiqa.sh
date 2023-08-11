#!/usr/bin/env bash

mkdir -p datasets/fiqa && cd datasets/fiqa
ir_datasets export beir/fiqa docs  > docs
ir_datasets export beir/fiqa docs --format jsonl \
  | python3 ../../data-preparation/prepend-passage.py \
  | python3 ../../data-preparation/add-title-if-missing.py \
  > docs.jsonl
gzip < docs.jsonl > passages.jsonl.gz
< docs.jsonl python3 ../../data-preparation/convert-for-feeding.py > feed.jsonl

ir_datasets export beir/fiqa/train qrels > train-qrels
ir_datasets export beir/fiqa/train queries > train-queries
ir_datasets export beir/fiqa/dev qrels > dev-qrels
ir_datasets export beir/fiqa/dev queries > dev-queries
ir_datasets export beir/fiqa/test qrels > test-qrels
ir_datasets export beir/fiqa/test queries > test-queries
