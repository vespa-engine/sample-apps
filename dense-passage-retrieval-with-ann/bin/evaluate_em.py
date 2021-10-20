#!/usr/bin/env python3

# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import requests
import sys
import numpy as np

from multiprocessing import Pool
sys.path.append("DPR")
from dpr.data.qa_validation import exact_match_score

def get_vespa_result(question, retriever_model):
  request_body = {
    'type': 'any',
    'hits': 10,
    'query': question,
    'retriever': retriever_model
  }
  url = endpoint + '/search/'
  response = requests.post(url, json=request_body)
  return response.json()

def get_result(q):
  question = q['question']
  print(question)
  answers = q['answer']
  result = get_vespa_result(question,retriever_model)
  try:
    hit = result['root']['children'][0]['fields']
    prediction = hit['prediction']
    context = hit['context']
    score = hit['prediction_score']
    reader_score = hit['reader_score']
    return (prediction,question,context,score,reader_score,answers)
  except:
    print("Failure" + json.dumps(result))
    return ('',answers)

input_file = sys.argv[1]
retriever_model = sys.argv[2]
endpoint = sys.argv[3]

questions = []
with open(input_file) as f:
  for line in f:
    question = json.loads(line)
    questions.append(question)
p = Pool(5)
results = p.map(get_result, questions)

em = []
predictions = []
for prediction,question,context, score, reader_score, answers in results:
  em_hit = max([exact_match_score(prediction, answer) for answer in answers])
  em.append(em_hit)
  prediction_entry = {
    "question": question,
    "prediction": prediction,
    "context": context,
    "span_score": score,
    "reader_score": reader_score,
    "answers": answers,
    "em": em_hit
  }
  predictions.append(prediction_entry)

with open("predictions.json", "w") as fp:
  json.dump(predictions, fp)

em_score = np.mean(em)
print("EM score = %.2f, n = %i" % (em_score*100, len(em)))

