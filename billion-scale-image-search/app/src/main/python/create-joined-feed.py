# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

import sys
import json
import numpy as np
import mmh3
import pyarrow.parquet as pq

def compute_hash(url, text):
  if url is None:
    url = ''

  if text is None:
    text = ''

  total = (url + text).encode("utf-8")
  return mmh3.hash64(total)[0]

def nan_handler(number):
  if np.isnan(number):
    return 0
  else :
    return number


parquet_file = pq.ParquetFile(sys.argv[1])

for batch in parquet_file.iter_batches(batch_size=50000):
  df = batch.to_pandas()
  for index, row in df.iterrows():
    url = row['url']
    caption = row['caption']
    id = compute_hash(url, caption)
    similarity = nan_handler(row['similarity'])
    similarity_scaled = min(int(100*similarity), 127)
    vector = np.array(row['vector'], dtype=np.float32)
    doc = {
      "put": "id:laion:image::%i" % id,
      "fields": {
        "url": row['url'],
        "caption": row['caption'],
        "nsfw": row['NSFW'],
        "similarity": similarity_scaled,
        "height": row['height'],
        "width": row['width'],
        "license": row['LICENSE'],
        "vector": {
          "values": vector.tolist()
        }
      }
    }
    print(json.dumps(doc))
