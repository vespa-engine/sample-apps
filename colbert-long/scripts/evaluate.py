# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import requests
from requests.adapters import HTTPAdapter, Retry
import argparse
import pandas


def search(query, query_id):
    global session
    global args
    global response_times
    global responses
    query_request = {
        'query': query, 
        'yql': 'select id from sources * where userQuery()', 
        'input.query(qt)': f'embed(colbert,"{query}")', 
        'hits' : args.hits, 
        'ranking': args.ranking,
        'presentation.timing' : True, 
        'language' : args.language, 
        'ranking.rerankCount': args.rank_count,
        'timeout' : '20s',
        'default-index': 'text'
    }
    try:
        response = session.post(args.endpoint, json=query_request,timeout=120)
    except:
        response = session.post(args.endpoint, json=query_request,timeout=120)
    if response.ok:
        json_result = response.json()
        time = json_result['timing']['searchtime']
        response_times.append(1000*time)
        root = json_result['root']
        total_count = root['fields']['totalCount']
        #print("query %s, latency %.4f, totalcount %i" % (query,time,total_count))
        if total_count > 0:
          pos = 1
          for hit in root['children']:
            id = hit['fields']['id']
            relevance = hit['relevance']
            if str(query_id) == str(id): 
              continue
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
      print("query request failed with " + str(response.json()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True) 
    parser.add_argument("--ranking", type=str, required=True) 
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--rank_count", type=int, default=100)
    parser.add_argument("--certificate", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--language", type=str, default="en")

    global args
    args = parser.parse_args()
    global session
    session = requests.Session()
    retries = Retry(total=20, connect=20,
      backoff_factor=0.3,
      status_forcelist=[ 500, 503, 504, 429 ]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))

    if args.certificate and args.key:
      session.cert = (args.certificate, args.key)
  
    global response_times 
    global responses
    responses = []
    response_times = []
    queries = []
    with open(args.dataset) as fp:
      for line in fp:
        line = line.strip()
        qid,query = line.split("\t")
        queries.append((qid,query))

    for id,q in queries: 
      search(q,id)
      
    df_result = pandas.DataFrame.from_records(responses)
    df_result.to_csv(args.ranking + ".run", index=False, header=False, sep=' ')
    import numpy as np
    print("Mean latency %.4f ms" % np.mean(response_times))

if __name__ == "__main__":
    main()

