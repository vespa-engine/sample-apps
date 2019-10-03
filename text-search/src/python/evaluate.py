#! /usr/bin/env python3

import sys
import csv
import os
import json
import urllib.request
import urllib.parse


RANK_PROFILE = sys.argv[1]
DATA_FOLDER = sys.argv[2] if len(sys.argv) > 2 else "data"
QUERY_RELEVANCE_FILE = os.path.join(DATA_FOLDER, "test-queries.tsv")
OUTPUT_METRICS_FILE = os.path.join(DATA_FOLDER, "test-output-" + RANK_PROFILE + ".tsv")


def vespa_search(query, rank_profile, hits=100, offset=0):
    """
    Query Vespa and retrieve results in JSON format.

    :param query: msmarco query.
    :param rank_profile: Ranking-profile used to sort documents
    :param hits: Number of hits to retrieve per page
    :param offset: Page to be retrieved
    :return: Vespa results in JSON format
    """
    url = "http://localhost:8080/search/?query={}&hits={}&offset={}&ranking={}&summary={}".format(
        urllib.parse.quote_plus(query),
        hits,
        offset,
        rank_profile,
        "minimal"
    )
    return json.loads(urllib.request.urlopen(url).read())


def parse_vespa_json(data):
    """
    Parse Vespa results to get necessary information.

    :param data: Vespa results in JSON format.
    :return: List with retrieved documents
    """
    ranking = []
    if "children" in data["root"]:
        ranking = [hit["fields"]["id"] for hit in data["root"]["children"] if "fields" in hit]
    return ranking


def compute_reciprocal_rank(ranking, relevant_id):
    """
    Compute reciprocal rank

    :param ranking: list with doc ids ordered by a ranking function
    :param relevant_id: relevant id
    :return: reciprocal rank
    """
    try:
        rank = ranking.index(relevant_id) + 1
        rr = 1 / rank
    except ValueError:
        rr = 0

    return rr


def main():

    with open(QUERY_RELEVANCE_FILE, encoding="utf8") as fin, \
         open(OUTPUT_METRICS_FILE, "w", encoding="utf8") as fout:
        reader = csv.reader(fin, delimiter="\t")
        rrs = []
        for row in reader:
            query = row[0].strip()
            relevant_id = row[1]

            vespa_result = vespa_search(query=query, rank_profile=RANK_PROFILE)
            ranking = parse_vespa_json(data=vespa_result)
            rr = compute_reciprocal_rank(ranking=ranking, relevant_id=relevant_id)
            rrs.append(rr)

            fout.write("{0}\n".format(rr))

        print("MRR: {}".format(sum(rrs) / len(rrs)))


if __name__ == "__main__":
    main()
