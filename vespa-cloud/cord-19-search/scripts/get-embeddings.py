#!/usr/bin/env python3
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import pandas
import sys
import json
from sentence_transformers import SentenceTransformer 

DATA_FILE = sys.argv[1]
MODEL_PATH = sys.argv[2]
model = SentenceTransformer(MODEL_PATH) 

def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]

titles = {}
abstracts = {}
with open(DATA_FILE, 'r') as f:
  for doc in json.load(f):
    id = doc['id']
    title = doc.get('title',None)
    if title:
      titles[title] = id 
    abstract = doc.get('abstract',None)
    if abstract:
      abstracts[abstract] = id 

title_encodings = {}
titles = list(titles.keys())
for batch in chunks(titles, 500):
  batch_encodings = model.encode(batch)
  for i in range(0,len(batch_encodings)):
    title_encodings[batch[i]] = batch_encodings[i].tolist()


abstract_encodings = {}
abstracts = list(abstracts.keys())
for batch in chunks(abstracts, 500):
  batch_encodings = model.encode(batch)
  for i in range(0, len(batch_encodings)):
    abstract_encodings[batch[i]] = batch_encodings[i].tolist()

print("\"id\",\"title_embedding\",\"abstract_embedding\"")
with open(DATA_FILE, 'r') as f:
  for doc in json.load(f):
    id = doc['id']
    title = doc.get('title',None)
    title_embedding = title_encodings.get(title,None)
    abstract = doc.get('abstract',None)
    abstract_embedding = abstract_encodings.get(abstract,None)
    print("\"%i\",\"%s\",\"%s\"" % (id,title_embedding,abstract_embedding))

