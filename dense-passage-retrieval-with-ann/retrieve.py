#!/usr/bin/env python3 
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json 
import requests
import sys
import gzip

from multiprocessing import Pool
sys.path.append("DPR")
from dpr.data.qa_validation import has_answer
from dpr.utils.tokenizers import SimpleTokenizer


def makeWeakAnd(query):
  words = query.split()
  terms = []
  for w in words:
    t = 'default contains \"%s\"' % w.replace("\"","")
    terms.append(t)
  weakAnd = '([{"targetNumHits":10}]weakAnd(%s))' % ','.join(terms)
  return weakAnd

def get_query(rank_profile, question):
  if rank_profile.startswith('sparse'):
    yql = 'select id, title, text from sources * where \
    %s;' % makeWeakAnd(question) 
  elif rank_profile.startswith('dense'):
    yql = 'select id, title, text from sources * where \
    ([{"targetNumHits":100, "hnsw.exploreAdditionalHits":1000}]nearestNeighbor(text_embedding,query_embedding));' 
  else:
    yql='select id, title, text from sources * where \
    ([{"targetNumHits":100, "hnsw.exploreAdditionalHits":1000}]nearestNeighbor(text_embedding,query_embedding)) \
    or %s;' % makeWeakAnd(question)
  return yql 

def get_vespa_result(question, query_embedding, rank_profile):
  yql = get_query(rank_profile, question)
  request_body = {
    'hits': 100,
    'timeout': 15,
    'ranking.softtimeout.enable': False,
    'type': 'any',
    'yql': yql,
    'query': question,
    'ranking': rank_profile,
    'ranking.features.query(query_embedding)': query_embedding,
  }
  url = endpoint + '/search/'
  response = requests.post(url, json=request_body) 
  return response.json()

def get_result(q):
  question = q['question']
  answers = q['answer']
  print("Retrieving Question '%s'" % question)
  embedding = q['embedding']
  embedding.append(0.0)
  result = get_vespa_result(question,embedding, rank_profile)
  n = result['root']['fields']['totalCount']
  if n < 100:
    print("question %s did not retrieve any hits" % question)
    qa = {
      "question": question,
      "answers": answers,
      "ctxs": [] 
    }
    return qa

  contexts = []
  for h in result['root']['children']:
    score = h['relevance']
    fields = h.get('fields',None)
    if not fields:
      continue
    title = fields['title']
    text = fields['text']
    id = fields['id']
    context = {
      "id": id, 
      "title": title,
      "text": text,
      "score": score,
      "has_answer": has_answer(answers, text, tokenizer, 'string') 
    }
    contexts.append(context)
  qa = {
    "question": question,
    "answers": answers,
    "ctxs": contexts 
  }
  return qa
    

tokenizer = SimpleTokenizer()

input_file = sys.argv[1]
rank_profile = sys.argv[2]
endpoint = sys.argv[3]

if input_file.endswith('gz'):
  with gzip.open(input_file) as f: 
    questions = json.load(f)
else:
  with open(input_file) as f: 
    questions = json.load(f)
p = Pool(10)
results = p.map(get_result, questions)

results = [r for r in results if r.get('question',None)]
with open("results-" + rank_profile + ".json", "w") as f:
  json.dump(results, f)

