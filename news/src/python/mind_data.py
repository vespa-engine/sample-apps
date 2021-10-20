#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import io
import sys
import json
import random
import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


class MindData:
    def __init__(self, train_news_file, train_impressions_file, valid_news_file, valid_impressions_file):
        self.user_map = {}    # Mapping from user_id to index, e.g. "U82271" -> 0, "U84185" -> 1 etc
        self.news_map = {}    # Mapping from news_id to index, e.g. "N27209" -> 0
        self.cat_map = {}     # Mapping from category to index, e.g. "health" -> 0
        self.sub_cat_map = {} # Mapping from subcategory to index, e.g. "medical" -> 0
        self.entity_map = {}  # Mapping from entity id to index, e.g. "Q212" -> 0  (WikidataId for "ukraine")
        self.news_content = {}
        self.train_impressions = []
        self.valid_impressions = []
        self.read_data(train_impressions_file, self.train_impressions)
        self.read_data(valid_impressions_file, self.valid_impressions)
        self.read_news(train_news_file)
        self.read_news(valid_news_file)

    def users(self):
        return self.user_map

    def news(self):
        return self.news_map

    def categories(self):
        return self.cat_map

    def subcategories(self):
        return self.sub_cat_map

    def entities(self):
        return self.entity_map

    def read_news(self, news_file):
        if not os.path.exists(news_file):
            print("{} not found.".format(news_file))
            sys.exit(1)
        print("Reading data from " + news_file)

        with io.open(news_file, "r", encoding="utf-8") as f:
            for line in f:
                news_data = line.split("\t")
                doc_id = news_data[0]
                category = news_data[1]
                subcategory = news_data[2]
                title = news_data[3]
                abstract = news_data[4]
                url = news_data[5]
                title_entities = json.loads(news_data[6]) if len(news_data) > 6 else []
                abstract_entities = json.loads(news_data[7]) if len(news_data) > 7 else []

                entities_to_keep = 2
                entity_ids = []
                self.add_entities(entity_ids, title_entities)
                self.add_entities(entity_ids, abstract_entities)
                entity_ids.extend([""] * entities_to_keep)

                self.news_content[self.lookup_news_index(doc_id)] = {
                    "news_id": doc_id,
                    "category_index": self.lookup_category_index(category),
                    "subcategory_index": self.lookup_subcategory_index(subcategory),
                    "entity_index": [ self.lookup_entity_index(id) for id in entity_ids[:entities_to_keep]]
                }

    def add_entities(self, entity_ids: list, entities: list):
        for entity in entities:
            entity_id = entity["WikidataId"]
            if entity_id not in entity_ids:
                entity_ids.append(entity_id)

    def read_data(self, impressions_file, impression_list):
        if not os.path.exists(impressions_file):
            print("{} not found.".format(impressions_file))
            sys.exit(1)
        print("Reading data from " + impressions_file)

        with io.open(impressions_file, "r", encoding="utf-8") as f:
            for line in f:
                impression_data = line.split("\t")
                impression_id = impression_data[0]
                user_id = impression_data[1]
                time = datetime.datetime.strptime(impression_data[2], "%m/%d/%Y %I:%M:%S %p")
                history = set(impression_data[3].split(" "))
                impressions = impression_data[4].split(" ")

                news_indices, labels = self.find_labels(impressions, history)

                impression_list.append({
                    "user_index": self.lookup_user_index(user_id),
                    "timestamp": time,
                    "news_indices": news_indices,
                    "labels": labels
                })

    def find_labels(self, impressions, history):
        news_indices = []
        labels = []
        for impression in impressions:
            news_id, label = impression.split("-")
            if news_id not in history:  # don't add news previously clicked (in history)
                news_indices.append(self.lookup_news_index(news_id))
                labels.append(int(label))
        return news_indices, labels

    def lookup_user_index(self, user_id):
        return self.lookup_index(user_id, self.user_map)

    def lookup_news_index(self, news_id):
        return self.lookup_index(news_id, self.news_map)

    def lookup_category_index(self, category):
        return self.lookup_index(category, self.cat_map)

    def lookup_subcategory_index(self, subcategory):
        return self.lookup_index(subcategory, self.sub_cat_map)

    def lookup_entity_index(self, entity_id):
        return self.lookup_index(entity_id, self.entity_map)

    def lookup_index(self, id, map):
        if id not in map:
            map[id] = len(map)
        return map[id]

    def sample_training_data(self, batch_size, num_negatives, prob=1.0):
        user_indices = []
        news_indices = []
        category_indices = []
        subcategory_indices = []
        entity_indices = []
        labels = []

        for impression in self.train_impressions:
            if random.random() <= prob:
                self.add_impressions(impression, user_indices, news_indices, category_indices, subcategory_indices, entity_indices, labels, num_negatives)

        data_set = TensorDataset(
                        torch.LongTensor(user_indices),
                        torch.LongTensor(news_indices),
                        torch.LongTensor(category_indices),
                        torch.LongTensor(subcategory_indices),
                        torch.LongTensor(entity_indices),
                        torch.FloatTensor(labels))
        generator = torch.Generator()
        random_sampler = RandomSampler(data_set, generator=generator)
        return DataLoader(data_set, batch_size=batch_size, sampler=random_sampler)

    def sample_valid_data(self, prob=1.0, train=False):
        data = []
        impressions = self.train_impressions if train else self.valid_impressions
        for impression in impressions:
            if random.random() <= prob:
                news_indices = impression["news_indices"]
                labels = impression["labels"]
                category_indices = [ self.news_content[index]["category_index"] for index in news_indices ]
                subcategory_indices = [ self.news_content[index]["subcategory_index"] for index in news_indices ]
                entity_indices = [ self.news_content[index]["entity_index"] for index in news_indices ]
                user_index = [ impression["user_index"] ] * len(labels)
                if sum(labels) > 0 and sum(labels) != len(labels):
                    data.append([
                        torch.LongTensor(user_index),
                        torch.LongTensor(news_indices),
                        torch.LongTensor(category_indices),
                        torch.LongTensor(subcategory_indices),
                        torch.LongTensor(entity_indices),
                        torch.FloatTensor(labels)
                    ])
        return data

    def add_impressions(self, impression, user_indices, news_indices, category_indices, subcategory_indices, entity_indices, labels, num_negatives):
        user_index = impression["user_index"]
        positive, negative = self.find_clicked(impression["news_indices"], impression["labels"])

        # add each positive label
        for pos_news_index in positive:
            user_indices.append(user_index)
            news_indices.append(pos_news_index)
            category_indices.append(self.news_content[pos_news_index]["category_index"])
            subcategory_indices.append(self.news_content[pos_news_index]["subcategory_index"])
            entity_indices.append(self.news_content[pos_news_index]["entity_index"])
            labels.append(1)

            # for each positive label, sample negative labels
            for neg_news_index in random.sample(negative, min(num_negatives, len(negative))):
                user_indices.append(user_index)
                news_indices.append(neg_news_index)
                category_indices.append(self.news_content[neg_news_index]["category_index"])
                subcategory_indices.append(self.news_content[neg_news_index]["subcategory_index"])
                entity_indices.append(self.news_content[neg_news_index]["entity_index"])
                labels.append(0)

    def find_clicked(self, news_ids, labels):
        clicked = []
        skipped = []
        for news_id, label in zip(news_ids, labels):
            if label > 0:
                clicked.append(news_id)
            else:
                skipped.append(news_id)
        return clicked, skipped

    def get_news_content_tensors(self):
        num_docs = len(self.news_map)
        news_indices = [0] * num_docs
        category_indices = [0] * num_docs
        subcategory_indices = [0] * num_docs
        entity_indices = [0] * num_docs

        for news_index, content in self.news_content.items():
            news_indices[news_index] = news_index
            category_indices[news_index] = content["category_index"]
            subcategory_indices[news_index] = content["subcategory_index"]
            entity_indices[news_index] = content["entity_index"]

        return torch.LongTensor(news_indices), \
               torch.LongTensor(category_indices), \
               torch.LongTensor(subcategory_indices),\
               torch.LongTensor(entity_indices),
