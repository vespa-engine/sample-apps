#! /usr/bin/env python3

import os
import sys
import csv
import json
import time
import urllib

from transformers import AutoTokenizer


data_dir = sys.argv[1] if len(sys.argv) > 1 else "msmarco"
queries_file = os.path.join(data_dir, "test-queries.tsv")

model_name = "nboost/pt-tinybert-msmarco"
sequence_length = 128
tokenizer = AutoTokenizer.from_pretrained(model_name)


def vespa_search(tokens, hits=10):
    tokens_str = "[" + ",".join( [ str(i) for i in tokens ]) + "]"
    url = "http://localhost:8080/search/?hits={}&timeout=600&query={}&ranking={}&ranking.features.query(input)={}".format(
              hits,
              "sddocname:msmarco",
              "bert",
              urllib.parse.quote_plus(tokens_str)
          )

    return json.loads(urllib.request.urlopen(url).read())


def main():
    view_max = 1
    with open(queries_file, encoding="utf8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            query = row[0].strip()
            print("Query: " + query)

            tokens = tokenizer.encode_plus(query, add_special_tokens=False, max_length=sequence_length, pad_to_max_length=True)["input_ids"]
            result = vespa_search(tokens)

            print(json.dumps(result, indent=2))

            view_max -= 1
            if view_max == 0:
                break



if __name__ == "__main__":
    main()
