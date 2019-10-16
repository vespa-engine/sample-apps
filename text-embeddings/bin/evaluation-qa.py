#!/usr/bin/env python3 

import requests
import json
import sys
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
import numpy as np

def mean_reciprocal_rank(rs):
  rs = (np.asarray(r).nonzero()[0] for r in rs)
  return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def handle_query(query,queryid,rank_profile):
  embedding = session.run(question_embeddings,feed_dict={question:[query]})['outputs'][0].tolist()
  json_request = {
    "query": "(query_id:>0 %s)" % query,
    "type": "any",
    "hits": 10,
    "recall": "+query_id:%s" % queryid,
    "timeout":20,
    "ranking.softtimeout.enable":"false",
    "ranking.features.query(tensor)": embedding,
    "ranking.profile": rank_profile 
  }
  r = requests.post('http://localhost:8080/search/', json=json_request)
  response = r.json()
  
  if response["root"]["fields"]["totalCount"] == 0:
    return [0]
  selected = []
  for hit in response["root"]["children"]:
    selected.append(hit["fields"]["is_selected"])
    print("Hit %.6f %i %s" % (hit["relevance"],hit["fields"]["is_selected"],hit["fields"]["passage_text"]))
  return selected 

print("Downloading universal sentence encoder multilingual-qa  which is  about 1GB which needs to be downloaded")
g = tf.Graph()
with g.as_default():
  module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/1")
  question = tf.compat.v1.placeholder(tf.string) 
  question_embeddings = module(
    dict(input=question),
    signature="question_encoder", as_dict=True)
  init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.compat.v1.Session(graph=g)
session.run(init_op)
print("Done creating TF session")

rank_profile_results = {
  "random":[], 
  "bm25":[], 
  "passage-semantic-similarity":[],
  "max-sentence-semantic-similarity":[],
}

for line in sys.stdin:
  line = line.strip()
  qid,query,query_type = line.split('\t')
  for profile in rank_profile_results:
    result = handle_query(query,qid,profile)
    print("%s %s %s" %(query,profile,str(result)))
    rank_profile_results[profile].append(result)

for profile in rank_profile_results.keys():
  print("Rank Profile '%s' MRR@10 Result: %.4f " % (profile,mean_reciprocal_rank(rank_profile_results[profile])))
