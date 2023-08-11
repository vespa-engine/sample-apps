#!/usr/bin/env bash

mkdir -p datasets/nq && cd datasets/nq

ir_datasets export beir/nq docs  > docs
ir_datasets export beir/nq docs --format jsonl \
  | python3 ../../data-preparation/prepend-passage.py \
  | python3 ../../data-preparation/add-title-if-missing.py \
  > docs.jsonl
gzip < docs.jsonl > passages.jsonl.gz
< docs.jsonl python3 ../../data-preparation/convert-for-feeding.py > feed.jsonl

ir_datasets export beir/nq qrels > qrels
ir_datasets export beir/nq queries > queries
