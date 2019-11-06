#! /usr/bin/env python3

import os
import sys
import requests
from pandas import DataFrame
from msmarco import load_msmarco_queries, load_msmarco_qrels, extract_querie_relevance

QUERIES_FILE_PATH = sys.argv[1]
RELEVANCE_FILE_PATH = sys.argv[2]
DATA_FOLDER = sys.argv[3]
RANK_PROFILE = sys.argv[4]
OUTPUT_FILE = os.path.join(
    DATA_FOLDER, "training_data_match_random_" + RANK_PROFILE + ".csv"
)
PROCESSED_QUERIES_FILE = os.path.join(
    DATA_FOLDER, "training_data_match_random_" + RANK_PROFILE + "_processed_queries.csv"
)


def create_request_specific_ids(query, rankprofile, doc_ids):
    body = {
        "yql": "select id, rankfeatures from sources * where (userInput(@userQuery));",
        "userQuery": query,
        "hits": len(doc_ids),
        "recall": "+(" + " ".join(["id:" + str(doc) for doc in doc_ids]) + ")",
        "timeout": "15s",
        "presentation.format": "json",
        "ranking": {"profile": rankprofile, "listFeatures": "true"},
    }
    return body


def create_request_top_hits(query, rankprofile, hits):

    body = {
        "yql": "select id, rankfeatures from sources * where (userInput(@userQuery));",
        "userQuery": query,
        "hits": hits,
        "timeout": "15s",
        "presentation.format": "json",
        "ranking": {"profile": rankprofile, "listFeatures": "true"},
    }
    return body


def get_features(url, body):

    r = requests.post(url, json=body)
    if r.status_code != requests.codes.ok:
        print("Bad response code for request: " + str(body))
        return {}
    result = r.json()
    hits = []
    if "children" in result["root"]:
        hits = result["root"]["children"]
    return hits


def annotate_data(hits, query_id, relevant_id):
    data = []
    for h in hits:
        rankfeatures = h["fields"]["rankfeatures"]
        rankfeatures.update({"docid": h["fields"]["id"]})
        rankfeatures.update({"qid": query_id})
        rankfeatures.update({"relevant": 1 if h["fields"]["id"] == relevant_id else 0})
        data.append(rankfeatures)
    return data


def build_dataset(url, query_relevance, rank_profile, number_random_sample):
    try:
        f_processed = open(PROCESSED_QUERIES_FILE)
        processed_queries = [int(x) for x in f_processed.readlines()]
        f_processed.close()
    except FileNotFoundError:
        processed_queries = []
    number_queries = len(query_relevance) - len(processed_queries)
    line = 0
    for qid, (query, relevant_id) in query_relevance.items():
        if int(qid) not in processed_queries:
            line += 1
            print("{0}/{1}".format(line, number_queries))

            relevant_id_request = create_request_specific_ids(
                query=query, rankprofile=rank_profile, doc_ids=[relevant_id]
            )
            hits = get_features(url=url, body=relevant_id_request)
            if len(hits) == 1 and hits[0]["fields"]["id"] == relevant_id:
                random_hits_request = create_request_top_hits(
                    query=query, rankprofile=rank_profile, hits=number_random_sample
                )
                hits.extend(get_features(url=url, body=random_hits_request))

                features = annotate_data(
                    hits=hits, query_id=qid, relevant_id=relevant_id
                )
                if os.path.isfile(OUTPUT_FILE):
                    DataFrame.from_records(features).to_csv(
                        OUTPUT_FILE, index=None, mode="a", header=False
                    )
                else:
                    DataFrame.from_records(features).to_csv(OUTPUT_FILE, index=None)
        with open(PROCESSED_QUERIES_FILE, "a") as f_processed:
            f_processed.write("{0}\n".format(qid))


def main():

    queries = load_msmarco_queries(queries_file_path=QUERIES_FILE_PATH)
    qrels = load_msmarco_qrels(relevance_file_path=RELEVANCE_FILE_PATH)
    query_relevance = extract_querie_relevance(qrels, queries)

    build_dataset(
        url="http://localhost:8080/search/",
        query_relevance=query_relevance,
        rank_profile=RANK_PROFILE,
        number_random_sample=10,
    )


if __name__ == "__main__":
    main()
