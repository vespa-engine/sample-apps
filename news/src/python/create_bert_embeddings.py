#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import io
import sys
import csv
import time

from transformers import logging, BertTokenizer, BertModel


logging.set_verbosity_error()

data_dir = sys.argv[1]
doc_type = "mind"

train_news_file = os.path.join(data_dir, "train", "news.tsv")
train_embedding_file = os.path.join(data_dir, "train", "news_embeddings.tsv")
dev_news_file = os.path.join(data_dir, "dev", "news.tsv")
dev_embedding_file = os.path.join(data_dir, "dev", "news_embeddings.tsv")

field_list = ["docid", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]


def create_embeddings(input_file_name, output_file_name):
    docids = set()
    titles = {}
    abstracts = {}
    categories = {}
    subcategories = {}

    # read
    with io.open(input_file_name, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=field_list)
        for line in reader:
            docid = line["docid"]
            docids.add(docid)
            titles[docid] = line["title"]
            abstracts[docid] = line["abstract"]
            categories[docid] = line["category"]
            subcategories[docid] = line["subcategory"]

    print("Read {} documents".format(len(docids)))

    # convert and write
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('google/bert_uncased_L-8_H-512_A-8')
    t = time.time()
    with open(output_file_name, "w") as f:
        for i, docid in enumerate(docids):
            title = titles[docid]
            abstract = abstracts[docid]

            f.write(docid)

            tokens = tokenizer(title, abstract, return_tensors="pt", max_length=100, truncation=True, padding=True)
            outputs = model(**tokens)
            write_embedding(f, outputs[0].tolist()[0][0])

            f.write("\n")

            if i % (len(docids)//100) == (len(docids)//100 - 1):
                print("Completed {} embeddings ({:.0f} %) [{:.2f} s]".format(i + 1, 100.0 *(i+1)/len(docids), time.time()-t))
                t = time.time()


def write_embedding(f, embedding):
    f.write("\t")
    for j, num in enumerate(embedding):
        if j > 0:
            f.write(",")
        f.write("%.4f" % num)


def main():
    create_embeddings(train_news_file, train_embedding_file)
    create_embeddings(dev_news_file, dev_embedding_file)


if __name__ == "__main__":
    main()


