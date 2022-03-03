#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import ir_datasets
import json
import numpy
import requests
import sys
import argparse

session = requests.Session()

# Fetch data from vespa routine
def get_result(query):
  request_body = {
    'hits': args.hits,
    'type': 'any',
    'presentation.summary':'id',
    'yql': "select id from sources * where userQuery()",
    'query': query,
    'queryProfile': args.query_profile
  }

  response = session.post(args.endpoint, json=request_body,timeout=25.05)
  if not response.ok:
    print("Failed request for query {}, response {}, response json {}".format(query, response,response.json()))
    return []

  result = response.json()
  if result['root']['fields']['totalCount'] == 0:
    return []
  docs = []
  for h in result['root']['children']:
    fields = h.get('fields',None)
    if not fields:
      continue
    docs.append(fields.get('id'))
  return docs

def eval():  
  queries = []
  dataset = ir_datasets.load('msmarco-passage/' + args.query_split + '/small')
  for query_id,text in dataset.queries_iter():
    queries.append((query_id,text))
  from tqdm import tqdm
  with open(args.run_file, "w") as fp:
    for id,query in tqdm(queries):
      docs = get_result(query)
      for i,d in enumerate(docs):
        if args.trec_format:
          fp.write("{}\tQ0\t{}\t{}\t{}\trun\n".format(id,d,i+1,1000-i))
        else:
          fp.write("{}\t{}\t{}\n".format(id,d,i+1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080/search/")
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--query_profile", type=str, required=True)
    parser.add_argument("--run_file", type=str, default="runfile") 
    parser.add_argument("--query_split", choices=["dev", "eval"], required=True)
    parser.add_argument('--trec_format', dest='trec_format', default=False, action='store_true')
    global args
    args = parser.parse_args()
    eval()

if __name__ == "__main__":
    main()

