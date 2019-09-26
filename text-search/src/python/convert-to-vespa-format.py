#! /usr/bin/env python3

import os
import io
import sys
import csv
import json

data_dir = sys.argv[1]
doc_type = sys.argv[2]
fields = sys.argv[3].split(",")

sample_offset_file = os.path.join(data_dir, "test-docs-offset.tsv")
docs_file = os.path.join(data_dir, "docs.tsv")
out_file = os.path.join(data_dir, "vespa.json")


def load_document_offsets():
    docoffset = {}
    with io.open(sample_offset_file, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, offset] in tsvreader:
            docoffset[docid] = int(offset)
    return docoffset


def main():
    document_offsets = load_document_offsets()

    docs = 0
    with io.open(docs_file, "r", encoding="utf-8") as f, open(out_file, "w") as out:
        out.write("[\n")
        for docid in document_offsets.keys():
            f.seek(document_offsets[docid])
            line = f.readline()
            line = line.strip()
            content = line.split("\t")

            found_docid = content[0]
            if found_docid != docid:
                continue  # dataset has some wrong lookup values
            if len(content) != len(fields):
                continue  # missing fields

            if docs > 0:
                out.write(",\n")
            docs += 1

            doc = { "put" : f"id:{doc_type}:{doc_type}::{docid}", "fields" : {} }
            for i, field in enumerate(fields):
                doc["fields"][field] = content[i]
            json.dump(doc, out)

        out.write("\n]\n")


if __name__ == "__main__":
    main()


