#! /usr/bin/env python3

import re
import os
import io
import csv
import sys
import random


min_relevant_docs_per_query = 1
data_dir = sys.argv[1]
queries_to_generate = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
documents_to_generate = int(sys.argv[3]) if len(sys.argv) > 3 else 100000

query_file = os.path.join(data_dir, "queries.tsv")
docs_file = os.path.join(data_dir, "docs.tsv")
offset_file = os.path.join(data_dir, "docs-offset.tsv")

sample_query_file = os.path.join(data_dir, "test-queries.tsv")
sample_offset_file = os.path.join(data_dir, "test-docs-offset.tsv")


def load_queries():
    queries = {}
    with io.open(query_file, "r", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [query, relevant_documents] in tsvreader:
            query = re.sub(r"[^\w ]", " ", query).lower()
            queries[len(queries)] = {
                "query": query,
                "docs": relevant_documents.split(","),
            }
    return queries


def load_document_offsets():
    docoffset = {}
    with io.open(offset_file, "r", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, offset] in tsvreader:
            docoffset[docid] = int(offset)
    return docoffset


# Only select queries that have more than a given number of relevance judgements
def select_queries(queries):
    qids = [ qid for qid in queries.keys() if len(queries[qid]["docs"]) >= min_relevant_docs_per_query ]
    qids = random.sample(qids, min(queries_to_generate, len(qids)))
    return set(qids)


# Sample documents_to_generate documents, with documents in queries guaranteed to be in the set
def select_documents(selected_query_ids, queries, offsets):
    selected_doc_ids = set()
    for qid in selected_query_ids:
        for docid in queries[qid]["docs"]:
            selected_doc_ids.add(docid)

    docids = [docid for docid in offsets.keys()]
    while len(selected_doc_ids) < min(len(docids), documents_to_generate):
        selected_doc_ids.add(random.choice(docids))

    return selected_doc_ids


def write_queries(qids, queries):
    print("Sampling {0} queries...".format(len(qids)))
    with open(sample_query_file, "w", encoding="utf8") as out:
        for qid in qids:
            relevant_documents = ",".join(queries[qid]["docs"])
            out.write("{0}\t{1}\n".format(queries[qid]["query"], relevant_documents))


def write_document_offsets(selected_doc_ids, offsets):
    print("Sampling {0} documents...".format(len(selected_doc_ids)))
    with open(sample_offset_file, "w", encoding="utf8") as out:
        for docid in selected_doc_ids:
            out.write("{0}\t{1}\n".format(docid, offsets[docid]))


def main():
    queries = load_queries()
    offsets = load_document_offsets()

    selected_query_ids = select_queries(queries)
    selected_doc_ids = select_documents(selected_query_ids, queries, offsets)

    write_queries(selected_query_ids, queries)
    write_document_offsets(selected_doc_ids, offsets)


if __name__ == "__main__":
    main()
