#!/usr/bin/env python3

import os
import csv
import json
import urllib.parse, urllib.request

data_dir = "msmarco"
queries_file = os.path.join(data_dir, "test-queries.tsv")


def vespa_search(query, hits=10):
    url = "http://localhost:8080/search/?hits={}&query={}".format(
              hits,
              urllib.parse.quote_plus(query)
          )
    print("Querying: " + url)
    return json.loads(urllib.request.urlopen(url).read())


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
