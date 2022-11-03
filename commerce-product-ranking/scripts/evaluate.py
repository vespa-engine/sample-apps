# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import numpy as np
import json
import os
import sys
import requests
from requests.adapters import HTTPAdapter, Retry
import argparse
import pandas

def search(query_group):
    ids = [] #the list of products we have labels for
    for (_, row) in query_group.iterrows():
      ids.append("id:{}".format(row['product_id'])) 
    recall = " ".join(ids)
    recall = "+({})".format(recall)
    query = row['query']
    query_id = row['query_id']
    query_request = {
        'yql': 'select id from product where userQuery() or ({targetHits:100, approximate:false}nearestNeighbor(embedding,query_embedding))',
        'query': query, 
        'input.query(query_embedding)': 'embed(transformer, "%s")'  % query, 
        'input.query(query_tokens)': 'embed(tokenizer, "%s")' %query,
        'ranking': args.ranking,
        'hits' : args.hits, 
        'timeout': '5s',
        'recall': recall ,
        'ranking.softtimeout.enable': 'false'
    }
    response = session.post(args.endpoint, json=query_request,timeout=120)
    if response.ok:
        json_result = response.json()
        root = json_result['root']
        total_count = root['fields']['totalCount']
        assert total_count == len(ids) #make sure we rank all 
        if total_count > 0:
          pos = 1
          for hit in root['children']:
            id = hit['fields']['id']
            relevance = hit['relevance']
            doc = {
              "query_id": query_id,
              "iteration": "Q0",
              "product_id": id,
              "position": pos,
              "score": relevance,
              "runid": args.ranking 
            }
            responses.append(doc)
            pos+=1
    else:
      print("request failed " + response.json())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True) 
    parser.add_argument("--ranking", type=str, required=True) 
    parser.add_argument("--example_file", type=str, required=True)
    parser.add_argument("--hits", type=int, default=400)
    parser.add_argument("--certificate", type=str)
    parser.add_argument("--key", type=str)

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
    df_examples = pandas.read_parquet(args.example_file)
    df_examples = df_examples[df_examples['split'] == "test"]
    df_examples = df_examples[df_examples['product_locale'] == "us"]
    df_examples = df_examples[df_examples['small_version'] == 1]
    df_examples.groupby("query_id").apply(search)
    df_result = pandas.DataFrame.from_records(responses)
    df_result.to_csv(args.ranking + ".run", index=False, header=False, sep=' ')

if __name__ == "__main__":
    main()

