# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter, Retry
import argparse
import ir_datasets
from sentence_transformers import CrossEncoder
from typing import NamedTuple
import json
import pandas as pd


def search(query, query_id):
    query_request = {
        "yql": 'select cord_uid, abstract from doc where {"grammar":"tokenize", "targetHits":200}nearestNeighbor(embedding, e)',
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
            "positives": {"doc_id": [], "score": []},
            "negatives": {"doc_id": [], "score": []},
        }

        known_positive_document_ids = qrels.get(query_id, [])
        data["positives"]["doc_id"].extend(known_positive_document_ids)
        known_positive_documents = dataset.docs.lookup_iter(known_positive_document_ids)
        data["positives"]["score"].extend(
            [
                float(model.predict([(query, document.text)])[0])
                for document in known_positive_documents
            ]
        )
        known_positive_score = data["positives"]["score"][0]

        for hit in json_result["root"]["children"]:
            cord_uid = hit["fields"]["cord_uid"]

            score = model.predict([(query, hit["fields"]["abstract"])])
            if score < known_positive_score - 3:
                data["negatives"]["doc_id"].append(cord_uid)
                data["negatives"]["score"].append(float(score[0]))

        if not data["positives"]["doc_id"] or not data["negatives"]["doc_id"]:
            return

        train_data.append(data)
    else:
        print("query request failed with " + str(response.json()))


class NFCorpusDoc(NamedTuple):
    doc_id: str
    text: str
    title: str
    url: str

    def default_text(self):
        return self.text


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
    parser.add_argument("--docs", type=str, required=True)
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

    global dataset
    dataset = ir_datasets.Dataset(
        ir_datasets.formats.TsvDocs(
            ir_datasets.util.LocalDownload(args.docs), doc_cls=NFCorpusDoc
        ),
        ir_datasets.formats.TsvQueries(ir_datasets.util.LocalDownload(args.queries)),
        ir_datasets.formats.TrecQrels(ir_datasets.util.LocalDownload(args.qrels), {}),
    )

    global model
    model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

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
