#! /usr/bin/env python3

import os
import sys
import gzip
import random
from math import floor


def main(input_dir, output_dir, percent_queries_train):

    with gzip.open(
        os.path.join(input_dir, "msmarco-doctrain-queries.tsv.gz"),
        "rt",
        encoding="utf-8",
    ) as f:
        queries = f.readlines()

    number_queries = len(queries)
    total_index = range(number_queries)

    #
    # Divide queries into train and test queries
    #
    queries_to_sample = floor(percent_queries_train * number_queries)
    train_indexes = random.sample(list(total_index), queries_to_sample)

    train_queries = [queries[x] for x in train_indexes]
    test_queries = [queries[x] for x in total_index if x not in train_indexes]

    with gzip.open(
        os.path.join(output_dir, "msmarco-doctrain-queries.tsv.gz"),
        "wt",
        encoding="utf-8",
    ) as f:
        for query in train_queries:
            f.write(query)

    with gzip.open(
        os.path.join(output_dir, "msmarco-doctest-queries.tsv.gz"),
        "wt",
        encoding="utf-8",
    ) as f:
        for query in test_queries:
            f.write(query)

    #
    # Divide train and test query relevance
    #
    train_qids = set([query.split("\t")[0] for query in train_queries])
    test_qids = set([query.split("\t")[0] for query in test_queries])

    with gzip.open(
        os.path.join(input_dir, "msmarco-doctrain-qrels.tsv.gz"), "rt", encoding="utf-8"
    ) as f:
        qrels = f.readlines()

    qrels = dict([(qrel.split("\t")[0], qrel) for qrel in qrels])

    #
    # Train qrels
    #
    train_qrels = [qrels[qid] for qid in train_qids]

    with gzip.open(
        os.path.join(output_dir, "msmarco-doctrain-qrels.tsv.gz"),
        "wt",
        encoding="utf-8",
    ) as f:
        for qrel in train_qrels:
            f.write(qrel)

    #
    # Test qrels
    #
    test_qrels = [qrels[qid] for qid in test_qids]

    with gzip.open(
        os.path.join(output_dir, "msmarco-doctest-qrels.tsv.gz"), "wt", encoding="utf-8"
    ) as f:
        for qrel in test_qrels:
            f.write(qrel)


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    percent_queries_train = float(sys.argv[3])
    main(
        input_dir=input_dir,
        output_dir=output_dir,
        percent_queries_train=percent_queries_train,
    )
