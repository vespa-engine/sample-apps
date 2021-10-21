#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import io
import sys
import numpy as np


data_dir = sys.argv[1] if len(sys.argv) > 1 else "../../mind/"
doc_type = "mind"

user_embeddings_file = os.path.join(data_dir, "user_embeddings.tsv")
news_embeddings_file = os.path.join(data_dir, "news_embeddings.tsv")
user_embeddings_vespa = os.path.join(data_dir, "vespa_user_embeddings.json")
news_embeddings_vespa = os.path.join(data_dir, "vespa_news_embeddings.json")


def read_embeddings(file_name):
    if not os.path.exists(file_name):
        print("{} not found.".format(file_name))
        sys.exit(1)
    print("Reading embeddings data from " + file_name)
    embeddings = {}
    with io.open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            id, vector = line.split("\t")
            embeddings[id] = np.array(vector.split(","), dtype=np.float32)
    return embeddings


def convert_user_embeddings():
    user_embeddings = read_embeddings(user_embeddings_file)
    prepend_value = 0.0  # prepend embedding with this value for conversion to euclidean distance
    with open(user_embeddings_vespa, "w") as out:
        out.write("[\n")
        for i, (user_id, embedding) in enumerate(user_embeddings.items()):
            if i > 0:
                out.write(",\n")
            embedding = np.concatenate(([prepend_value], embedding))
            embedding_str = ",".join(["%.6f" % v for v in embedding])
            out.write('{"put": "id:user:user::%s", ' % user_id +
                      '"fields": {"user_id":"%s", ' % user_id +
                      '"embedding": {"values": [%s]} }}' % embedding_str)
        out.write("\n]\n")


def convert_news_embeddings():
    news_embeddings = read_embeddings(news_embeddings_file)
    max_embedding_length = max([np.linalg.norm(vector) for vector in news_embeddings.values()]) + 1e-6
    with open(news_embeddings_vespa, "w") as out:
        out.write("[\n")
        for i, (news_id, embedding) in enumerate(news_embeddings.items()):
            if i > 0:
                out.write(",\n")
            embedding_length = np.linalg.norm(embedding)
            prepend_value = np.sqrt(np.power(max_embedding_length, 2) - np.power(embedding_length, 2))
            embedding = np.concatenate(([prepend_value], embedding))
            embedding_str = ",".join(["%.6f" % v for v in embedding])
            out.write('{"update": "id:news:news::%s", ' % news_id +
                      '"fields": {"embedding": {"assign": { "values": [%s]} }}}' % embedding_str)
        out.write("\n]\n")


def main():
    convert_user_embeddings()
    convert_news_embeddings()


if __name__ == "__main__":
    main()


