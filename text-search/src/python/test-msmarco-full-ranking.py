#! /usr/bin/env python3

import sys
import gzip
import os
import csv
import re
from time import time
from requests import post

RANK_PROFILE = sys.argv[1]
RUN_ID = sys.argv[2]
DATA_FOLDER = sys.argv[3]
QUERIES_FILE = sys.argv[4]
RELEVANCE_FILE = sys.argv[5] if len(sys.argv) > 5 else None

QUERIES_FILE_PATH = os.path.join(DATA_FOLDER, QUERIES_FILE)
OUTPUT_FILE = os.path.join(DATA_FOLDER, RUN_ID + "_" + RANK_PROFILE + ".txt")
if RELEVANCE_FILE:
    RELEVANCE_FILE_PATH = os.path.join(DATA_FOLDER, RELEVANCE_FILE)
    OUTPUT_METRIC = os.path.join(DATA_FOLDER, RUN_ID + "_" + RANK_PROFILE + "_rr.tsv")
    OUTPUT_METRIC_SUMMARY = os.path.join(
        DATA_FOLDER, RUN_ID + "_" + RANK_PROFILE + "_metric_summary.tsv"
    )


def load_msmarco_queries(queries_file_path):
    print("Loading queries...")
    queries = {}
    with gzip.open(queries_file_path, "rt", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [queryid, query] in tsvreader:
            query = re.sub(r"[^\w ]", " ", query).lower()
            queries[queryid] = query.strip()
    return queries


def load_msmarco_qrels(relevance_file_path):
    """
    Map query id to relevant doc ids

    :return: For each queryid, the list of positive docids is qrel[queryid]
    """
    print("Loading query relevance judgements...")
    qrel = {}
    with gzip.open(relevance_file_path, "rt", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [queryid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if queryid in qrel:
                qrel[queryid].add(docid)
            else:
                qrel[queryid] = set([docid])
    return qrel


def extract_querie_relevance(qrel, query_strings):
    """Create output file with query id, query string and relevant doc"""
    print("Extracting {0} queries...".format(len(qrel)))
    query_relevance = {}
    for qid in qrel.keys():
        relevant_documents = ",".join([docid for docid in qrel[qid]])
        query_relevance[qid] = (query_strings[qid], relevant_documents)
    return query_relevance


def vespa_search(query, rank_profile, hits=1000, offset=0):
    """
    Query Vespa and retrieve results in JSON format.

    :param query: msmarco query.
    :param rank_profile: Ranking-profile used to sort documents
    :param hits: Number of hits to retrieve per page
    :param offset: Page to be retrieved
    :return: Vespa results in JSON format
    """
    # "yql": 'select * from sources * where ([{"grammar": "any"}]userInput(@userQuery));',
    body = {
        "yql": "select * from sources * where (userInput(@userQuery));",
        "userQuery": query,
        "hits": hits,
        "offset": offset,
        "ranking": rank_profile,
        "summary": "minimal",
        "presentation.format": "json",
    }
    r = post("http://localhost:8080/search/", json=body)
    return r.json()


def parse_vespa_json(data):
    """
    Parse Vespa results to get necessary information.

    :param data: Vespa results in JSON format.
    :return: List with retrieved documents together with their relevance score.
    """
    ranking = []
    if "children" in data["root"]:
        ranking = [
            (hit["fields"]["id"], hit["relevance"])
            for hit in data["root"]["children"]
            if "fields" in hit
        ]
    return ranking


def evaluate(query_relevance):
    print("Computing evaluations ...")
    with open(OUTPUT_METRIC, "w", encoding="utf8") as f_metric:
        with open(OUTPUT_METRIC_SUMMARY, "w", encoding="utf8") as f_metric_summary:
            number_queries = 0
            total_rr = 0
            start_time = time()
            for qid, (query, relevant_id) in query_relevance.items():
                rr = 0
                vespa_result = vespa_search(query=query, rank_profile=RANK_PROFILE)
                ranking = parse_vespa_json(data=vespa_result)
                for rank, hit in enumerate(ranking):
                    if hit[0] == relevant_id:
                        rr = 1 / (rank + 1)
                f_metric.write("{0}\t{1}\n".format(qid, rr))
                total_rr += rr
                number_queries += 1
            execution_time = time() - start_time
            f_metric_summary.write("name\tvalue\n")
            f_metric_summary.write(
                "{0}\t{1}\n".format("number_queries", number_queries)
            )
            f_metric_summary.write(
                "{0}\t{1}\n".format("qps", number_queries / execution_time)
            )
            f_metric_summary.write(
                "{0}\t{1}\n".format("mrr", total_rr / number_queries)
            )


def generate_search_results(queries):
    print("Generate search results ...")
    for qid, query in queries.items():
        vespa_result = vespa_search(query=query, rank_profile=RANK_PROFILE)
        ranking = parse_vespa_json(data=vespa_result)
        for rank, hit in enumerate(ranking):
            with open(OUTPUT_FILE, "a", encoding="utf8") as fout:
                fout.write(
                    "{0} {1} {2} {3} {4} {5}\n".format(
                        qid, "Q0", hit[0], rank, hit[1], RUN_ID
                    )
                )


def main():
    queries = load_msmarco_queries(queries_file_path=QUERIES_FILE_PATH)
    if RELEVANCE_FILE:
        qrels = load_msmarco_qrels(relevance_file_path=RELEVANCE_FILE_PATH)
        query_relevance = extract_querie_relevance(qrels, queries)
        evaluate(query_relevance)
    else:
        generate_search_results(queries)


if __name__ == "__main__":
    main()
