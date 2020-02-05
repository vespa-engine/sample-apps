#! /usr/bin/env python3

import os
import sys
import requests
from pandas import DataFrame
import tensorflow_hub as hub
from msmarco import load_msmarco_queries, load_msmarco_qrels, extract_querie_relevance
from embedding import create_document_embedding


def create_request_specific_ids(
    query, rankprofile, doc_ids, word2vec_embedding, gse_embedding
):
    body = {
        "yql": "select id, rankfeatures from sources * where (userInput(@userQuery));",
        "userQuery": query,
        "hits": len(doc_ids),
        "recall": "+(" + " ".join(["id:" + str(doc) for doc in doc_ids]) + ")",
        "timeout": "15s",
        "presentation.format": "json",
        "ranking.features.query(tensor)": str(word2vec_embedding),
        "ranking.features.query(tensor_gse)": str(gse_embedding),
        "ranking": {"profile": rankprofile, "listFeatures": "true"},
    }
    return body


def create_request_top_hits(
    query, rankprofile, hits, word2vec_embedding, gse_embedding
):

    body = {
        "yql": "select id, rankfeatures from sources * where (userInput(@userQuery));",
        "userQuery": query,
        "hits": hits,
        "timeout": "15s",
        "presentation.format": "json",
        "ranking.features.query(tensor)": str(word2vec_embedding),
        "ranking.features.query(tensor_gse)": str(gse_embedding),
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


def load_processed_queries(file_path):
    try:
        f_processed = open(file_path)
        processed_queries = [int(x) for x in f_processed.readlines()]
        f_processed.close()
    except FileNotFoundError:
        processed_queries = []
    return processed_queries


def build_dataset(url, query_relevance, rank_profile, number_random_sample):
    word2vec_model = hub.load(
        "https://tfhub.dev/google/Wiki-words-500-with-normalization/2"
    )
    gse_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    processed_queries = load_processed_queries(file_path=PROCESSED_QUERIES_FILE)
    number_queries = len(query_relevance) - len(processed_queries)
    line = 0
    for qid, (query, relevant_id) in query_relevance.items():
        if int(qid) not in processed_queries:
            line += 1
            print("{0}/{1}".format(line, number_queries))
            word2vec_embedding = create_document_embedding(
                text=query, model=word2vec_model, normalize=True
            )
            gse_embedding = create_document_embedding(
                text=query, model=gse_model, normalize=True
            )
            relevant_id_request = create_request_specific_ids(
                query=query,
                rankprofile=rank_profile,
                doc_ids=[relevant_id],
                word2vec_embedding=word2vec_embedding,
                gse_embedding=gse_embedding,
            )
            hits = get_features(url=url, body=relevant_id_request)
            if len(hits) == 1 and hits[0]["fields"]["id"] == relevant_id:
                random_hits_request = create_request_top_hits(
                    query=query,
                    rankprofile=rank_profile,
                    hits=number_random_sample,
                    word2vec_embedding=word2vec_embedding,
                    gse_embedding=gse_embedding,
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
        number_random_sample=NUMBER_RANDOM_SAMPLE,
    )


if __name__ == "__main__":
    DATA_FOLDER = sys.argv[1]
    RANK_PROFILE = sys.argv[2]
    NUMBER_RANDOM_SAMPLE = sys.argv[3]

    QUERIES_FILE_PATH = os.path.join(DATA_FOLDER, "msmarco-doctrain-queries.tsv.gz")
    RELEVANCE_FILE_PATH = os.path.join(DATA_FOLDER, "msmarco-doctrain-qrels.tsv.gz")

    OUTPUT_FILE = os.path.join(
        DATA_FOLDER,
        "training_data_"
        + RANK_PROFILE
        + "_"
        + NUMBER_RANDOM_SAMPLE
        + "_random_samples.csv",
    )
    PROCESSED_QUERIES_FILE = os.path.join(
        DATA_FOLDER,
        "training_data_"
        + RANK_PROFILE
        + "_"
        + NUMBER_RANDOM_SAMPLE
        + "_random_samples_processed_queries.csv",
    )

    main()
