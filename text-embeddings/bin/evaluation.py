#!/usr/bin/env python3 

import requests
import json
import sys
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def mean_reciprocal_rank(rs):
  rs = (np.asarray(r).nonzero()[0] for r in rs)
  return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def handle_query(query,queryid,rank_profile):
  embedding = session.run(embeddings,feed_dict={text_sentences:[query]})[0].tolist()
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
  return selected 

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
module = hub.Module(module_url)
text_sentences = tf.compat.v1.placeholder(tf.string)
embeddings = module(text_sentences)

session = tf.compat.v1.Session() 
session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

rank_profile_results = {
  "random":[], 
  "passage-semantic-similarity":[],
  "max-sentence-semantic-similarity":[],
  "bm25":[],
  "nativeRank":[],
  "nativeRank-and-max-sentence-linear":[]
}

for line in sys.stdin:
  line = line.strip()
  qid,query,query_type = line.split('\t')
  for profile in rank_profile_results:
    result = handle_query(query,qid,profile)
    rank_profile_results[profile].append(result)
  
for profile in rank_profile_results.keys():
  print("Rank Profile '%s' MRR@10 Result: %.4f " % (profile,mean_reciprocal_rank(rank_profile_results[profile])))


