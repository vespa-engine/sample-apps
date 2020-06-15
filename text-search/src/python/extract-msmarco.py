#! /usr/bin/env python3

import re
import os
import csv
import gzip


download_dir = os.path.join("msmarco", "download")
query_file = os.path.join("msmarco", "queries.tsv")
docs_file = os.path.join("msmarco", "docs.tsv")
offset_file = os.path.join("msmarco", "docs-offset.tsv")

# if data is not downloaded - use sample data
if not os.path.isdir(download_dir):
    download_dir = os.path.join("msmarco", "sample")


# The query string for each queryid is querystring[queryid]
def load_msmarco_queries(qrel):
    print("Loading queries...")
    queries = {}
    with gzip.open(os.path.join(download_dir, "msmarco-doctrain-queries.tsv.gz"), "rt", encoding="utf8", ) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [queryid, query] in tsvreader:
            query = re.sub(r"[^\w ]", " ", query).lower()
            queries[queryid] = query
    return queries


# In the corpus tsv, each docid occurs at offset docoffset[docid]
def load_msmarco_document_lookup_table():
    print("Loading document lookup table...")
    docoffset = {}
    with gzip.open(os.path.join(download_dir, "msmarco-docs-lookup.tsv.gz"), "rt", encoding="utf8" ) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)
    return docoffset


# For each queryid, the list of positive docids is qrel[queryid]
def load_msmarco_qrels():
    print("Loading query relevance judgements...")
    qrel = {}
    with gzip.open(os.path.join(download_dir, "msmarco-doctrain-qrels.tsv.gz"), "rt", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for row in tsvreader:
            if len(row) == 4:
                [queryid, _, docid, rel] = row
            elif len(row) == 1:
                [queryid, _, docid, rel] = row[0].split(' ')
            assert rel == "1"
            if queryid in qrel:
                qrel[queryid].add(docid)
            else:
                qrel[queryid] = set([docid])
    return qrel


def getcontent(docid, f, docoffset):
    try:
        f.seek(docoffset[docid])
        line = f.readline()
        line = line.rstrip()
        [found_docid, url, title, body] = line.split("\t")
        assert docid == found_docid, f"Looking for {docid}, found {found_docid}"
        title = re.sub(r"(?<=[a-z\.])(?=[A-Z])", " ", title )  # Fix "corrosion resistanceInstall with" etc
        title = re.sub(r"\W", " ", title)  # Remove whitespace
        body = re.sub(r"(?<=[a-z\.])(?=[A-Z])", " ", body)
        body = re.sub(r"\W", " ", body)
        return "{0}\t{1}\t{2}\t{3}\n".format(docid, url, title, body)
    except Exception as e:
        pass
    return ""


# TODO: speed up, takes about an hour...
def extract_documents(docoffset):
    print("Extracting {0} documents...".format(len(docoffset.keys())))
    with open(os.path.join(download_dir, "msmarco-docs.tsv"), encoding="utf8" ) as f, \
         open(docs_file, "w", encoding="utf8") as out, \
         open(offset_file, "w", encoding="utf8") as offset:
        for docid in docoffset.keys():
            offset.write("{0}\t{1}\n".format(docid, out.tell()))
            out.write(getcontent(docid, f, docoffset))


def extract_queries(qrel, query_strings, docoffset):
    print("Extracting {0} queries...".format(len(qrel)))
    with open(query_file, "w", encoding="utf8") as out:
        for qid in qrel.keys():
            relevant_documents = ",".join( [docid for docid in qrel[qid] if docid in docoffset] )
            out.write("{0}\t{1}\n".format(query_strings[qid], relevant_documents))


def main():
    query_relevance = load_msmarco_qrels()
    query_strings = load_msmarco_queries(query_relevance)
    document_offset = load_msmarco_document_lookup_table()

    extract_documents(document_offset)
    extract_queries(query_relevance, query_strings, document_offset)


if __name__ == "__main__":
    main()
