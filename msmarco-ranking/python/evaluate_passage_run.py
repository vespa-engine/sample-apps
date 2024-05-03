#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import ir_datasets
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry


query_models =  {
  'bm25': {
    'yql': 'select id from passage where userQuery()',
    'ranking': 'bm25',
    'query': '{query}',
    'input.query(qt)':'embed(colbert, "{query}")'
  },
   'bm25-colbert': {
    'yql': 'select id from passage where userQuery()',
    'ranking': 'bm25-colbert',
    'input.query(qt)':'embed(colbert, "{query}")',
    'query': '{query}',
    'ranking.rerankCount': 100
  },
  "e5": {
    'yql': 'select id from passage where {targetHits: 10, hnsw.exploreAdditionalHits:100}nearestNeighbor(e5, q)',
    'input.query(q)': 'embed(e5, "{query}")',
    'ranking': 'e5',
  },
  "e5-colbert": {
    'yql': 'select id from passage where {targetHits: 100, hnsw.exploreAdditionalHits:100}nearestNeighbor(e5, q)',
    'input.query(q)': 'embed(e5, "{query}")',
    'input.query(qt)':'embed(colbert, "{query}")',
    'ranking': 'e5-colbert',
    'ranking.rerankCount': 100
  },
  "e5-colbert-cross-encoder-rrf": {
    'yql': 'select id from passage where {targetHits: 100,hnsw.exploreAdditionalHits:100}nearestNeighbor(e5, q)',
    'input.query(q)': 'embed(e5, "{query}")',
    'input.query(qt)':'embed(colbert, "{query}")',
    'input.query(query_token_ids)':'embed(tokenizer, "{query}")',
    'ranking': 'e5-colbert-cross-encoder-rrf',
    'ranking.rerankCount': 100, 
    'ranking.globalPhase.rerankCount': 20
  }
}

def format_query(template, query):
    return template.format(query=query)

# Fetch data from Vespa 
def get_result(query):
    global session, args
    params = query_models[args.model]
    formatted_params = dict()
    for key, value in params.items():
        if isinstance(value, str) and "{query}" in value:
            formatted_params[key] = format_query(value, query)
        else:
            formatted_params[key] = value
    request_body = {
        'hits': args.hits,
        'timeout': 10,
        'ranking.softtimeout.enable': 'false',
        **formatted_params
    }
    
    response = session.post(args.endpoint, json=request_body, timeout=10.0)
    if not response.ok:
        print("Failed request for query {}, response {}, response json {}".format(query, response, response.json()))
        return []

    result = response.json()
    if result['root']['fields']['totalCount'] == 0:
        return []
    docs = []
    for h in result['root']['children']:
        fields = h.get('fields', None)
        if not fields:
            continue
        docs.append(fields.get('id'))
    return docs

def eval_query(query):
    global args, session
    id, text = query
    docs = get_result(text)
    result_list = []
    for i, d in enumerate(docs):
        if args.trec_format:
            result_list.append("{}\tQ0\t{}\t{}\t{}\trun".format(id, d, i + 1, 1000 - i))
        else:
            result_list.append("{}\t{}\t{}".format(id, d, i + 1))
    return result_list

def eval():
    queries = []
    dataset = ir_datasets.load('msmarco-passage/' + args.query_split + '/small')
    for query_id, text in dataset.queries_iter():
        queries.append((query_id, text))
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(eval_query, queries), total=len(queries), desc="Fetching Results", leave=False))
    
    with open(args.run_file, "w") as fp:
        for result_list in results:
            for line in result_list:
                fp.write("{}\n".format(line))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080/search/")
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--model", choices=list(query_models.keys()), required=True)
    parser.add_argument("--run_file", type=str, default="runfile")
    parser.add_argument("--query_split", choices=["dev", "eval"], required=True)
    parser.add_argument('--trec_format', dest='trec_format', default=False, action='store_true')
    parser.add_argument("--certificate", type=str)
    parser.add_argument("--key", type=str)

    global args
    args = parser.parse_args()
    global session
    session = requests.Session()
    retries = Retry(total=5, connect=2,
      backoff_factor=0.3,
      status_forcelist=[ 500, 503, 504, 429 ]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))

    if args.certificate and args.key:
      session.cert = (args.certificate, args.key)
    
    eval()

if __name__ == "__main__":
    main()
