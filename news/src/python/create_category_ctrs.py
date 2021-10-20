#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import io
import sys
import csv
import json

from collections import defaultdict


data_dir = sys.argv[1] if len(sys.argv) > 1 else "../../mind/"
doc_type = "news"

train_news_file = os.path.join(data_dir, "train", "news.tsv")
train_impressions_file = os.path.join(data_dir, "train", "behaviors.tsv")
dev_news_file = os.path.join(data_dir, "dev", "news.tsv")
dev_impressions_file = os.path.join(data_dir, "dev", "behaviors.tsv")
global_ctr_file = os.path.join(data_dir, "global_category_ctr.json")
news_category_ctr_update_file = os.path.join(data_dir, "news_category_ctr_update.json")


def read_impressions_file(file_name, click_map, news_map):
    if not os.path.exists(file_name):
        print("{} not found.".format(file_name))
        sys.exit(1)
    print("Reading impressions data from " + file_name)

    with io.open(file_name, "r", encoding="utf-8") as f:
        field_list = ["id", "user_id", "timestamp", "history", "impressions"]
        reader = csv.DictReader(f, delimiter="\t", fieldnames=field_list)
        for line in reader:
            for impression in line["impressions"].split(" "):
                news_id, label = impression.split("-")
                if news_id in news_map:
                    category = news_map[news_id]
                    click_map[category]["impressions"] += 1
                    if label == "1":
                        click_map[category]["clicks"] += 1


def read_content_file(file_name, news_map):
    if not os.path.exists(file_name):
        print("{} not found.".format(file_name))
        sys.exit(1)
    print("Reading news data from " + file_name)

    with io.open(file_name, "r", encoding="utf-8") as f:
        field_list = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        reader = csv.DictReader(f, delimiter="\t", fieldnames=field_list)
        for line in reader:
            news_map[line["news_id"]] = line["category"]


def write_global_ctr_document(category_ctr_map):
    tensor_cells = []
    for category, ctr_data in category_ctr_map.items():
        category_ctr = ctr_data["clicks"] / ctr_data["impressions"] if ctr_data["impressions"] > 0 else 0.0
        cell = {"address":{"category":category}, "value": category_ctr}
        tensor_cells.append(cell)
    ctrs_tensor = {"ctrs": {"cells": tensor_cells}}

    global_ctr_document = {"put": "id:category_ctr:category_ctr::global", "fields": ctrs_tensor}
    with open(global_ctr_file, "w") as out:
        json.dump(global_ctr_document, out)


def write_news_category_ctr_update(news_to_cat_map):
    global_ctr_document_id = "id:category_ctr:category_ctr::global"
    with open(news_category_ctr_update_file, "w") as out:
        out.write("[\n")
        for i, (news_id, category) in enumerate(news_to_cat_map.items()):
            if i > 0:
                out.write(",\n")
            category_tensor = '{"cells": [{"address":{"category":"%s"}, "value": 1.0}]}' % category
            out.write('{"update": "id:news:news::%s", ' % news_id +
                      '"fields": {' +
                          '"category_ctr_ref": {"assign": "%s"}, ' % global_ctr_document_id +
                          '"category_tensor": {"assign": %s}' % category_tensor +
                      '}}')
        out.write("\n]\n")


def main():
    news_to_cat_map = {}
    read_content_file(train_news_file, news_to_cat_map)
    read_content_file(dev_news_file, news_to_cat_map)

    category_ctr_map = defaultdict(lambda: {"clicks": 0, "impressions": 0})
    read_impressions_file(train_impressions_file, category_ctr_map, news_to_cat_map)
    read_impressions_file(dev_impressions_file, category_ctr_map, news_to_cat_map)

    write_global_ctr_document(category_ctr_map)
    write_news_category_ctr_update(news_to_cat_map)


if __name__ == "__main__":
    main()



