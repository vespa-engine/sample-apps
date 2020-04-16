#!/usr/bin/env python3
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import pandas
import sys
import json


DATA_FILE = sys.argv[1]
EMBEDDINGS = sys.argv[2]

embeddings = pandas.read_csv(EMBEDDINGS)

docs = [] 
with open(DATA_FILE, 'r') as f:
  for doc in json.load(f):
    id = doc['id']
    row =  embeddings[embeddings.id == id]
    if not row.empty:
      title_embedding = eval(row.iloc[0]['title_embedding'])
      abstract_embedding = eval(row.iloc[0]['abstract_embedding'])
      if title_embedding:
        doc['title_embedding'] = {
          "values": title_embedding
        }
      if abstract_embedding:
        doc['abstract_embedding'] = {
          "values": abstract_embedding
        }
    docs.append(doc)

print(json.dumps(docs,indent=2))
