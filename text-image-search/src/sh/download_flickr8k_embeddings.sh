#!/usr/bin/env sh

mkdir -p embeddings
curl -L -o embeddings/flickr-8k-clip-embeddings.jsonl.zst https://data.vespa-cloud.com/sample-apps-data/flickr-8k-clip-embeddings.jsonl.zst
