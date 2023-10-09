# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter, Retry
import argparse
import ir_datasets
import json


def search(query, query_id):
    query_request = {
        "yql": 'select cord_uid from doc where {"grammar":"tokenize", "targetHits":200}nearestNeighbor(embedding, e)',
        "input.query(e)": f"embed({query})",
        "ranking": args.ranking,
        "cross-rerank": args.reranking,
        "cross-rerank-count": args.reranking_count,
        "hits": args.hits,
        "language": "en",
        "input.query(title_weight)": args.title_weight,
        "timeout": "20s",
    }
    try:
        response = session.post(args.endpoint, json=query_request, timeout=120)
    except:
        response = session.post(args.endpoint, json=query_request, timeout=120)
    if response.ok:
        json_result = response.json()

        data = {
            "query_id": query_id,
            "query": query,
            "positives": [],
            "negatives": [],
        }

        positive_documents = qrels.get(query_id, [])
        data["positives"].extend(positive_documents)

        for hit in json_result["root"]["children"]:
            cord_uid = hit["fields"]["cord_uid"]
            if cord_uid not in positive_documents:
                data["negatives"].append(cord_uid)

        if not data["positives"] or not data["negatives"]:
            return
        train_data.append(data)
    else:
        print("query request failed with " + str(response.json()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--ranking", type=str, required=True)
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--certificate", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--query_field", type=str, default="text")
    parser.add_argument("--title_weight", type=float, default=0.50)
    parser.add_argument("--reranking", action="store_true", default=False)
    parser.add_argument("--reranking_count", type=int, default=20)
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    global args
    args = parser.parse_args()
    global session
    session = requests.Session()
    retries = Retry(
        total=20, connect=20, backoff_factor=0.3, status_forcelist=[500, 503, 504, 429]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    if args.certificate and args.key:
        session.cert = (args.certificate, args.key)

    dataset = ir_datasets.create_dataset(
        queries_tsv=args.queries, qrels_trec=args.qrels
    )

    global qrels
    qrels = {
        query_id: documents.keys()
        for query_id, documents in dataset.qrels_dict().items()
    }
    global train_data
    train_data = []

    for q in tqdm(dataset.queries):
        query_text = f"query: {q.text}"
        search(query_text, q.query_id)

    with open(args.output_file, "w") as file:
        for data in train_data:
            file.write(f"{json.dumps(data)}\n")


if __name__ == "__main__":
    main()
