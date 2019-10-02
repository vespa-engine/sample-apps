#! /usr/bin/env python3

import os
import sys
import gzip
import random

queries_to_sample = 100
docs_to_sample = 1000

download_dir = sys.argv[1]
sample_dir = os.path.join("msmarco", "sample")

with gzip.open(os.path.join(download_dir, "msmarco-doctrain-queries.tsv.gz"), "rt", encoding="utf-8") as f:
   queries = f.readlines()

sampled_queries = random.sample(queries, queries_to_sample)
qids = set( [ query.split("\t")[0] for query in sampled_queries ] )

with gzip.open(os.path.join(sample_dir, "msmarco-doctrain-queries.tsv.gz"), "wt", encoding="utf-8") as f:
  for query in sampled_queries:
      f.write(query)

with gzip.open(os.path.join(download_dir, "msmarco-doctrain-qrels.tsv.gz"), "rt", encoding="utf-8") as f:
   qrels = f.readlines()

qrels = dict( [ (qrel.split("\t")[0],qrel) for qrel in qrels ] )
relevant_qrels = [ qrels[qid] for qid in qids ]
relevant_docids = [ qrel.split("\t")[2] for qrel in relevant_qrels ]

with gzip.open(os.path.join(sample_dir, "msmarco-doctrain-qrels.tsv.gz"), "wt", encoding="utf-8") as f:
    for qrel in relevant_qrels:
        f.write(qrel)

with gzip.open(os.path.join(download_dir, "msmarco-docs-lookup.tsv.gz"), "rt", encoding="utf8" ) as f:
    docoffset = f.readlines()

docoffset = dict( [ (offset.split("\t")[0], int(offset.split("\t")[2])) for offset in docoffset ] )
docids = docoffset.keys()
sampled_docids = relevant_docids
while len(sampled_docids) < docs_to_sample:
    sampled_docids.extend(random.sample(docids, docs_to_sample - len(relevant_docids)))

with open(os.path.join(download_dir, "msmarco-docs.tsv"), "r", encoding="utf8") as f, \
     open(os.path.join(sample_dir, "msmarco-docs.tsv"), "w", encoding="utf8") as out, \
     gzip.open(os.path.join(sample_dir, "msmarco-docs-lookup.tsv.gz"), "wt", encoding="utf8") as offset:
    for docid in sampled_docids:
        offset.write("{0}\t0\t{1}\n".format(docid, out.tell()))
        f.seek(docoffset[docid])
        out.write(f.readline())
