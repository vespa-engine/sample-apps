# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

import pandas as pd
import sys
import json
import numpy as np
import mmh3
import binascii

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
  

df = pd.read_parquet(sys.argv[1])
vectors = np.load(sys.argv[2], mmap_mode='r')

for index, row in df.iterrows():
  url = row['url']
  caption = row['caption']
  id = compute_hash(url, caption)
  similarity = nan_handler(row['similarity'])
  similarity_scaled = min(int(100*similarity), 127)
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
        "values": vectors[index].astype(np.float32).tolist() 
      }
    }
  }
  print(json.dumps(doc))

