#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import io
import sys
import csv
import json
import random
import urllib.parse 
import requests

from metrics import ndcg, mrr, group_auc


data_dir = sys.argv[1] if len(sys.argv) > 1 else "../../mind/"
sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
doc_type = "news"

train_impressions_file = os.path.join(data_dir, "train", "behaviors.tsv")
valid_impressions_file = os.path.join(data_dir, "dev", "behaviors.tsv")


def read_impressions_file(file_name):
    impressions = []
    if not os.path.exists(file_name):
        print("{} not found.".format(file_name))
        sys.exit(1)
    print("Reading impressions data from " + file_name)

    with io.open(file_name, "r", encoding="utf-8") as f:
        field_list = ["id", "user_id", "timestamp", "history", "impressions"]
        reader = csv.DictReader(f, delimiter="\t", fieldnames=field_list)
        for line in reader:
            user_id = line["user_id"]
            impression = {
                "user_id": user_id,
                "news_ids": [],
                "labels": []
            }
            for i in line["impressions"].split(" "):
                news_id, label = i.split("-")
                impression["news_ids"].append(news_id)
                impression["labels"].append(int(label))
            impressions.append(impression)
    return impressions


def parse_embedding(hit_json):
    return hit_json["fields"]["embedding"]["values"]

def query_user_embedding(user_id):
    yql = 'select * from sources user where user_id contains "{}"'.format(user_id)
    url = 'http://localhost:8080/search/?yql={}&hits=1'.format(urllib.parse.quote_plus(yql))  
    result = requests.get(url).json()
    return parse_embedding(result["root"]["children"][0])

def query_news(user_vector, news_ids):
    hits = len(news_ids)
    nn_annotations = [
        'targetHits:{}'.format(hits)
    ]
    nn_annotations = '{' + ','.join(nn_annotations) + '}'
    nn_search = "({}nearestNeighbor(embedding, user_embedding))".format(nn_annotations)

    news_id_filter = [ 'news_id contains "{}"'.format(i) for i in news_ids ]
    news_id_filter = " OR ".join(news_id_filter)

    data = {
        'hits': hits,
        'yql': 'select * from sources news where {} AND ({})'.format(nn_search, news_id_filter),
        'ranking.features.query(user_embedding)': str(user_vector),
        'ranking.profile': 'recommendation',
        'timeout': 10
    }
    return requests.post('http://localhost:8080/search/', json=data).json()

def find_hit(hits, news_id):
    for child in hits["root"]["children"]:
        if child["fields"]["news_id"] == news_id:
            return child
    return None


def predictions(hits, news_ids):
    preds = []
    for news_id in news_ids:
        hit = find_hit(hits, news_id)
        relevance = hit["relevance"]
        preds.append(relevance)
    return preds


def calc_impression(impression):
    user_id = impression["user_id"]
    user_vector = query_user_embedding(user_id)
    result = query_news(user_vector, impression["news_ids"])
    preds = predictions(result, impression["news_ids"])
    labels = impression["labels"]

    return preds, labels


def calc_metrics(file):
    all_predictions = []
    all_labels = []
    impressions = read_impressions_file(file)
    if sample_size > 0:
        impressions = random.sample(impressions, sample_size)
    for i, impression in enumerate(impressions):
        preds, labels = calc_impression(impression)
        all_predictions.append(preds)
        all_labels.append(labels)
        if i % 100 == 99:
            print("Completed {} / {}".format(i+1,len(impressions)))

    metrics = {
        "auc": group_auc(all_predictions, all_labels),
        "mrr": mrr(all_predictions, all_labels),
        "ndcg@5": ndcg(all_predictions, all_labels, 5),
        "ndcg@10": ndcg(all_predictions, all_labels, 10)
    }
    return metrics


train_metrics = calc_metrics(train_impressions_file)
valid_metrics = calc_metrics(valid_impressions_file)

print("Train: " + str(train_metrics))
print("Valid: " + str(valid_metrics))
