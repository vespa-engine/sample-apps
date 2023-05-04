#!/usr/bin/env python3

import os
import csv
import json
import requests

data_dir = "msmarco"
queries_file = os.path.join(data_dir, "test-queries.tsv")


def vespa_search(query, hits=10):
    request_body = {
        "query": query,
        "yql": "select * from msmarco where userQuery()",
        "input.query(q)": "embed(%s)".format(query),
        "ranking": "transformer"
    }
    url = "http://localhost:8080/search/"
    print("Querying: " + url)
    response = requests.post(url, json=request_body)
    return response.json()


def main():
    view_max = 1
    with open(queries_file, encoding="utf8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            query = row[0].strip()
            print("Query: " + query)

            result = vespa_search(query)
            print(json.dumps(result, indent=2))

            view_max -= 1
            if view_max == 0:
                break


if __name__ == "__main__":
    main()
