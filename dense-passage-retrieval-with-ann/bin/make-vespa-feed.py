#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import csv
import sys
import numpy as np
import json

sys.path.append("DPR")

from dense_retriever import iterate_encoded_files

def load_passages(file):
  docs = dict()
  with open(file) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
    for row in reader:
      id = row['id']
      text = row['text']
      title = row['title']
      docs[id] = (text,title)
  return docs

docs = load_passages(sys.argv[1])
vector_files = sys.argv[2:]
phi = 275.26935 #pre-computed phi for transformation from inner dot product space to euclidean space.

# Write out an empty query document for the question encoder
doc = { "put":"id:query:query::1", "fields": {} }
json.dump(doc,sys.stdout)
sys.stdout.write('\n')

# Write all wikipedia articles
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

