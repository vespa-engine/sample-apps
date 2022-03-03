#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import ir_datasets
import json
import numpy
import requests
from multiprocessing import Pool
import sys
import argparse

def get_result(query):
  request_body = {
    'hits': args.hits,
    'type': 'any',
    'presentation.summary':'id',
    'yql': "select id from sources * where userQuery()",
    'query': query,
    'retriever': args.retriever,
    'ranking.profile': args.rank_profile,
    'wand.field': args.wand_field,
    'wand.hits': args.wand_hits,
    'phase.count': args.phase_count,
    'restrict': 'doc',
    'searchChain': 'docranking'
  }
  response = requests.post(args.endpoint, json=request_body,timeout=25.05)
  if not response.ok:
    print("Failed request for query {}, response {}, response json {}".format(query, response,response.json()))
    return [0]

  result = response.json()
  if result['root']['fields']['totalCount'] == 0:
    return [0]
  docs = []
  for h in result['root']['children']:
    fields = h.get('fields',None)
    if not fields:
      continue
    docs.append(fields.get('id'))
  return docs

def do():  
  dataset = ir_datasets.load('msmarco-document/' + args.query_split) 
  lookup = ir_datasets.wrappers.DocstoreWrapper(dataset).queries_store()
  with open(args.run_file, "w") as fp:
    for id,query in dataset.queries_iter():
      docs = get_result(query)
      for i,d in enumerate(docs):
        fp.write("{}\t{}\t{}\n".format(id,d,i+1))
     
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_profile", type=str, required=True)
    parser.add_argument("--retriever", choices=["sparse"], required=True)
    parser.add_argument("--query_split", choices=["dev", "eval"], required=True)
    parser.add_argument("--wand_field", type=str, default="default") 
    parser.add_argument("--wand_hits", type=int, default=100) 
    parser.add_argument("--phase_count", type=int, default=1000) 
    parser.add_argument("--run_file", type=str, default="runfile")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080/search/")
    parser.add_argument("--hits", type=int, default=100)
    global args
    args = parser.parse_args()
    do()

if __name__ == "__main__":
    main()
