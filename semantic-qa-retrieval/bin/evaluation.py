#!/usr/bin/env python3 
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import requests
import json
import sys
import numpy as np


def mean_reciprocal_rank(rs):
  rs = (np.asarray(r).nonzero()[0] for r in rs)
  return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def recall_at(result,n,number_relevant):
  if number_relevant == 0:
    return 0.0
  return sum(result[0:n])/number_relevant

def handle_query_grouped(query, question_id, rank_profile, embedding, document_type, dataset="squad", hits=100):
  json_request = {
    "query": query,
    "type": "any",
    "yql": 'select * from sources sentence where ({"targetNumHits":100, "hnsw.exploreAdditionalHits":100}nearestNeighbor(sentence_embedding,query_embedding))\
 or userQuery() | all(group(context_id) max(%i) each(each(output(summary())) as(sentences)) as(paragraphs))' % hits,
    "hits": 0,
    "ranking.features.query(query_embedding)": embedding,
    "ranking.profile": rank_profile
  }
  r = requests.post('http://localhost:8080/search/', json=json_request)
  r.raise_for_status()
  response = r.json()

  if response["root"]["fields"]["totalCount"] == 0:
    return [0]
  ranked = []
  grouped_results = response["root"]["children"][0]["children"][0]["children"]
  for group in grouped_results:
    questions = [] #questions answered by paragraph
    context_id = group["value"]
    for hit in group['children'][0]["children"]:
      fields = hit["fields"]
      if "questions" in fields:
        questions.extend(fields["questions"])
    if question_id in questions:
      ranked.append(1)
    else:
      ranked.append(0)
  return ranked


def handle_query(query, question_id, rank_profile, embedding, document_type, dataset="squad", hits=100):
  json_request = {
    "query": query,
    "type": "any",
    "yql": 'select * from sources sentence where ({"targetNumHits":100, "hnsw.exploreAdditionalHits":100}nearestNeighbor(sentence_embedding,query_embedding)) or userQuery()',
    "hits": hits,
    "ranking.features.query(query_embedding)": embedding,
    "ranking.profile": rank_profile 
  }
  r = requests.post('http://localhost:8080/search/', json=json_request)
  r.raise_for_status() 
  response = r.json()

  if response["root"]["fields"]["totalCount"] == 0:
    return [0]
  ranked = []
  for hit in response["root"]["children"]:
    fields = hit['fields']
    questions = set() 
    if "questions" in fields:
      questions = set(fields['questions'])
    if question_id in questions:
      ranked.append(1) 
    else:
      ranked.append(0) 
  return ranked 


def run_evaluation(document_type, rank_profile, dataset="squad", sentence_grouping=False,hits=100):
  results = []
  recall_at_1 = []
  recall_at_5 = []
  recall_at_10 = []

  for qid, query, n_relevant,embedding in queries: 
    if sentence_grouping: 
      result = handle_query_grouped(query, qid, rank_profile, embedding, document_type, dataset, hits)
    else:
      result = handle_query(query, qid, rank_profile, embedding, document_type, dataset,hits)
    results.append(result)
    recall_at_1.append( recall_at(result, 1, n_relevant) )
    recall_at_5.append( recall_at(result, 5, n_relevant) )
    recall_at_10.append( recall_at(result, 10, n_relevant) )

  n = len(queries) 

  print("Profile '%s', doc='%s', dataset='%s',   MRR@100  %.4f" % (rank_profile, document_type, dataset, mean_reciprocal_rank(results)))
  print("Profile '%s', doc='%s', dataset='%s',   R@1 %.4f" % (rank_profile, document_type, dataset, sum(recall_at_1)/n))
  print("Profile '%s', doc='%s', dataset='%s',   R@5 %.4f" % (rank_profile, document_type, dataset, sum(recall_at_5)/n))
  print("Profile '%s', doc='%s', dataset='%s',   R@10 %.4f" % (rank_profile, document_type, dataset, sum(recall_at_10)/n))

queries = []
for line in sys.stdin:
  line = line.strip()
  qid,query,n_relevant,embedding = line.split('\t')
  queries.append((int(qid), query, int(n_relevant),eval(embedding)))

print("Start query evaluation for %i queries" % len(queries))
print("Sentence retrieval metrics:")
run_evaluation("sentence","sentence-semantic-similarity","squad",False)
print("Paragraph retrieval metrics:")
run_evaluation("sentence","sentence-semantic-similarity","squad",True)
