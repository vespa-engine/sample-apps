# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import json
import os
import sys
import requests
from requests.adapters import HTTPAdapter, Retry
import argparse
import pandas

def search(query_group):
    for (_, row) in query_group.iterrows():
      query = row['query']
      query_id = row['query_id']
    query_request = {
        'yql': 'select doc_id from doc where ({"grammar":"tokenize", "targetHits":200}userInput(@query)) or ({targetHits:200}nearestNeighbor(embedding, q))',
        'query': query, 
        'input.query(q)': f'embed({query})', 
        'ranking': args.ranking,
        'hits' : args.hits, 
        'language': args.language,
        'ranking.softtimeout.enable': 'false',
        'timeout': '25s'
    }
    response = session.post(args.endpoint, json=query_request,timeout=180)
    if response.ok:
        json_result = response.json()
        root = json_result['root']
        total_count = root['fields']['totalCount']
        if total_count > 0:
          pos = 1
          for hit in root['children']:
            id = hit['fields']['doc_id']
            relevance = hit['relevance']
            doc = {
              "query_id": query_id,
              "iteration": "Q0",
              "doc_id": id,
              "position": pos,
              "score": relevance,
              "runid": args.ranking 
            }
            responses.append(doc)
            pos+=1
    else:
      print("request failed " + str(response.json()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True) 
    parser.add_argument("--ranking", type=str, required=True) 
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--certificate", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--language", type=str, default="en")

    global args
    args = parser.parse_args()
    global session
    session = requests.Session()
    retries = Retry(total=10, connect=10,
      backoff_factor=0.3,
      status_forcelist=[ 500, 503, 504, 429 ]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))

    if args.certificate and args.key:
      session.cert = (args.certificate, args.key)
    
    global responses
    responses = []  
   
    df_examples = pandas.read_csv(args.query_file, sep='\t', names=["query_id", "query"])
    df_examples.groupby("query_id").apply(search)
    df_result = pandas.DataFrame.from_records(responses)
    df_result.to_csv(args.ranking + ".run", index=False, header=False, sep=' ')

if __name__ == "__main__":
    main()

