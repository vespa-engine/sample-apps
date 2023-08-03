# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import json
import os
import sys
import requests
from requests.adapters import HTTPAdapter, Retry
import argparse
import pandas
import ir_datasets


def search(query, query_id):
    query_request = {
        'yql': 'select cord_uid from cloudbeir where {"grammar":"tokenize", "targetHits":200}userInput(@query)',
        'query': query,
        'ranking': args.ranking,
        'cross-rerank': args.reranking,
        'cross-rerank-count': args.reranking_count,
        'hits': args.hits,
        'language': 'en',
        'input.query(title_weight)': args.title_weight,
        'timeout': '20s'
    }
    try:
        print("FAEN")
        response = session.post(args.endpoint, json=query_request, timeout=30)
    except Exception as e:  # Catch any exception
        print("An error occurred:", e)
    if response.ok:
        print("ok")
        json_result = response.json()
        root = json_result['root']
        total_count = root['fields']['totalCount']
        if total_count > 0:
            pos = 1
            for hit in root['children']:
                id = hit['fields']['cord_uid']
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
                pos += 1
    else:
        print("query request failed with error code " + str(response.status_code))
        print("aborting...")
        exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--ranking", type=str, required=True)
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--certificate", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--query_field", type=str, default="text")
    parser.add_argument("--title_weight", type=float, default=0.50)
    parser.add_argument('--reranking', action='store_true', default=False)
    parser.add_argument('--reranking_count', type=int, default=20)
    parser.add_argument("--dataset", type=str, default="beir/trec-covid")

    global args
    args = parser.parse_args()
    global session
    session = requests.Session()
    retries = Retry(total=1, connect=20,
                    backoff_factor=0.3,
                    status_forcelist=[500, 503, 504, 429]
                    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))

    if args.certificate and args.key:
        print("Trying to authorize")
        session.cert = (args.certificate, args.key)
        print("Done")
    dataset = ir_datasets.load(args.dataset)
    global responses
    responses = []
    for q in dataset.queries_iter():
        print("Query: " + q.text)
        if args.query_field == "query":
            print("yabba")
            search(q.query, q.query_id)
        else:
            print("dabba")
            search(q.text, q.query_id)
    df_result = pandas.DataFrame.from_records(responses)
    df_result.to_csv(args.ranking + ".run", index=False, header=False, sep=' ')


if __name__ == "__main__":
    print("Working Directory: " + os.getcwd())
    main()
