import gzip
import csv
import re


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


def load_corpus_doc_ids(doc_offset_file):
    # load all doc_ids that are in the corpus
    with open(doc_offset_file, "rt", encoding="utf8") as f_docs:
        docoffset = f_docs.readlines()
    return [offset.split("\t")[0] for offset in docoffset]


def extract_querie_relevance(qrel, query_strings):
    """Create output file with query id, query string and relevant doc"""
    print("Extracting {0} queries...".format(len(qrel)))
    query_relevance = {}
    for qid in qrel.keys():
        relevant_documents = ",".join([docid for docid in qrel[qid]])
        query_relevance[qid] = (query_strings[qid], relevant_documents)
    return query_relevance
