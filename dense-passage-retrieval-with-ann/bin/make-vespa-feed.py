#!/usr/bin/env python3 
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import numpy as np
import json
sys.path.append("DPR")
from dense_retriever import iterate_encoded_files, load_passages

docs = load_passages(sys.argv[1])
vector_files = sys.argv[2:]
phi = 275.26935 #pre-computed phi for transformation from inner dot product space to euclidean space. 

for i, item in enumerate(iterate_encoded_files(vector_files)):
  db_id, doc_vector = item
  norm = (doc_vector ** 2).sum()
  aux_dim = np.sqrt(phi - norm) 
  l2_vector = np.hstack((doc_vector, aux_dim))
  passage_text, title = docs[db_id]
  doc = {
    "put":"id:wiki:wiki::%s" % db_id,
    "fields": {
      "title": title,
      "text": passage_text,
      "id": int(db_id),
      "text_embedding": {
        "values": l2_vector.tolist()
      }
    }
  }
  json.dump(doc,sys.stdout)
  sys.stdout.write('\n')
