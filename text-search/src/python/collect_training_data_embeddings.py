#! /usr/bin/env python3

import os
import sys
import requests
from pandas import DataFrame
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from msmarco import load_msmarco_queries, load_msmarco_qrels, extract_querie_relevance
from embedding import create_document_embedding
from experiments import create_vespa_body_request


def create_request_specific_ids(
    query,
    rankprofile,
    grammar_operator,
    ann_operator,
    embedding_type,
    doc_ids,
    embedding_vector,
):
    """
    Create request to retrieve document with specific id.


    :param query: Query string.
    :param rankprofile: Rank profile name as it stands on the .sd file
    :param grammar_operator: The grammar used for the query term. Either 'AND', 'OR', 'weakAND' or None
    :param ann_operator: What is the ann operator used. Either 'title', 'body', 'title_body' or None
    :param embedding_type: What is the embedding type used, either 'word2vec', 'gse' or 'bert'.
    :param doc_ids: List with document ids that we want retrieved.
    :return: body to be send as POST request to Vespa
    """

    body = create_vespa_body_request(
        query=query,
        parsed_rank_profile=rankprofile,
        grammar_operator=grammar_operator,
        ann_operator=ann_operator,
        embedding_type=embedding_type,
        hits=len(doc_ids),
        offset=0,
        summary=None,
        embedding_vector=embedding_vector,
        tracelevel=None,
    )

    body.update(
        {
            "recall": "+(" + " ".join(["id:" + str(doc) for doc in doc_ids]) + ")",
            "timeout": "15s",
        }
    )
    return body


def create_request_top_hits(
    query,
    rankprofile,
    grammar_operator,
    ann_operator,
    embedding_type,
    hits,
    embedding_vector,
):
    """
    Create request to retrieve top hits according to a rank-pprofile and match phase.

    :param query: Query string.
    :param rankprofile: Rank profile name as it stands on the .sd file
    :param grammar_operator: The grammar used for the query term. Either 'AND', 'OR', 'weakAND' or None
    :param ann_operator: What is the ann operator used. Either 'title', 'body', 'title_body' or None
    :param embedding_type: What is the embedding type used, either 'word2vec', 'gse' or 'bert'.
    :param doc_ids: List with document ids that we want retrieved.
    :return: body to be send as POST request to Vespa
    """

    body = create_vespa_body_request(
        query=query,
        parsed_rank_profile=rankprofile,
        grammar_operator=grammar_operator,
        ann_operator=ann_operator,
        embedding_type=embedding_type,
        hits=hits,
        offset=0,
        summary=None,
        embedding_vector=embedding_vector,
        tracelevel=None,
    )

    body.update({"timeout": "15s"})
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


def retrieve_model(model_type):
    if model_type == "word2vec":
        return {
            "model": hub.load(
                "https://tfhub.dev/google/Wiki-words-500-with-normalization/2"
            ),
            "model_source": "tf_hub",
        }
    elif model_type == "gse":
        return {
            "model": hub.load("https://tfhub.dev/google/universal-sentence-encoder/4"),
            "model_source": "tf_hub",
        }
    elif model_type == "bert":
        return {
            "model": SentenceTransformer("distilbert-base-nli-stsb-mean-tokens"),
            "model_source": "bert",
        }


def build_dataset(
    url,
    query_relevance,
    number_random_sample,
    grammar_operator,
    ann_operator,
    embedding_type,
    rank_profile,
):
    model_info = retrieve_model(model_type=embedding_type)
    processed_queries = load_processed_queries(file_path=PROCESSED_QUERIES_FILE)
    number_queries = len(query_relevance) - len(processed_queries)
    line = 0
    for qid, (query, relevant_id) in query_relevance.items():
        if int(qid) not in processed_queries:
            line += 1
            print("{0}/{1}".format(line, number_queries))
            embedding_vector = create_document_embedding(
                text=query,
                model=model_info["model"],
                model_source=model_info["model_source"],
                normalize=True,
            )
            relevant_id_request = create_request_specific_ids(
                query=query,
                rankprofile=rank_profile,
                grammar_operator=grammar_operator,
                ann_operator=ann_operator,
                embedding_type=embedding_type,
                doc_ids=[relevant_id],
                embedding_vector=embedding_vector,
            )
            hits = get_features(url=url, body=relevant_id_request)
            if len(hits) == 1 and hits[0]["fields"]["id"] == relevant_id:
                random_hits_request = create_request_top_hits(
                    query=query,
                    rankprofile=rank_profile,
                    grammar_operator=grammar_operator,
                    ann_operator=ann_operator,
                    embedding_type=embedding_type,
                    hits=number_random_sample,
                    embedding_vector=embedding_vector,
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
        number_random_sample=NUMBER_RANDOM_SAMPLE,
        grammar_operator=GRAMMAR_OPERATOR,
        ann_operator=ANN_OPERATOR,
        embedding_type=EMBEDDING_TYPE,
        rank_profile=RANK_PROFILE,
    )


if __name__ == "__main__":
    DATA_FOLDER = sys.argv[1]
    NUMBER_RANDOM_SAMPLE = sys.argv[2]
    GRAMMAR_OPERATOR = sys.argv[3]
    ANN_OPERATOR = sys.argv[4]
    EMBEDDING_TYPE = sys.argv[5]
    RANK_PROFILE = sys.argv[6]

    QUERIES_FILE_PATH = os.path.join(DATA_FOLDER, "msmarco-doctrain-queries.tsv.gz")
    RELEVANCE_FILE_PATH = os.path.join(DATA_FOLDER, "msmarco-doctrain-qrels.tsv.gz")

    OUTPUT_FILE = os.path.join(
        DATA_FOLDER,
        "training_data_"
        + GRAMMAR_OPERATOR
        + "_"
        + ANN_OPERATOR
        + "_"
        + EMBEDDING_TYPE
        + "_"
        + RANK_PROFILE
        + "_"
        + NUMBER_RANDOM_SAMPLE
        + "_random_samples.csv",
    )
    PROCESSED_QUERIES_FILE = os.path.join(
        DATA_FOLDER,
        "training_data_"
        + GRAMMAR_OPERATOR
        + "_"
        + ANN_OPERATOR
        + "_"
        + EMBEDDING_TYPE
        + "_"
        + RANK_PROFILE
        + "_"
        + NUMBER_RANDOM_SAMPLE
        + "_random_samples_processed_queries.csv",
    )

    main()
