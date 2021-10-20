#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import io
import os
import sys
import numpy
import random
import torch
import numpy as np

from mind_data import MindData
from metrics import ndcg, mrr, group_auc


data_dir = sys.argv[1] if len(sys.argv) > 1 else "../../mind/"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15

train_news_file = os.path.join(data_dir, "train", "news.tsv")
valid_news_file = os.path.join(data_dir, "dev", "news.tsv")
train_impressions_file = os.path.join(data_dir, "train", "behaviors.tsv")
valid_impressions_file = os.path.join(data_dir, "dev", "behaviors.tsv")
train_embeddings_file = os.path.join(data_dir, "train", "news_embeddings.tsv")
valid_embeddings_file = os.path.join(data_dir, "dev", "news_embeddings.tsv")

# hyperparameters
embedding_size = 50
negative_sample_size = 4
batch_size = 128

adam_lr = 1e-3  # 5e-4
l2_regularization = 1e-5  # 0.01


class ContentBasedModel(torch.nn.Module):
    def __init__(self, num_users, num_news, num_categories, num_subcategories, num_entities, embedding_size, bert_embeddings):
        super(ContentBasedModel, self).__init__()

        # Embedding tables for category variables
        self.user_embeddings = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_size)
        self.news_embeddings = torch.nn.Embedding(num_embeddings=num_news, embedding_dim=embedding_size)
        self.cat_embeddings = torch.nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_size)
        self.sub_cat_embeddings = torch.nn.Embedding(num_embeddings=num_subcategories, embedding_dim=embedding_size)
        self.entity_embeddings = torch.nn.Embedding(num_embeddings=num_entities, embedding_dim=embedding_size)

        # Pretrained BERT embeddings
        self.news_bert_embeddings = torch.nn.Embedding.from_pretrained(bert_embeddings, freeze=True)

        # Linear transformation from BERT size to embedding size (512 -> 50)
        self.news_bert_transform = torch.nn.Linear(bert_embeddings.shape[1], embedding_size)

        # Linear transformation from concatenation of category, subcategory, entity and BERT embedding
        self.news_content_transform = torch.nn.Linear(in_features=embedding_size*5, out_features=embedding_size)

    def get_user_embeddings(self, users):
        return self.user_embeddings(users)

    def get_news_embeddings(self, items, categories, subcategories, entities):
        # Transform BERT representation to a shorter embedding
        bert_embeddings = self.news_bert_embeddings(items)
        bert_embeddings = self.news_bert_transform(bert_embeddings)
        bert_embeddings = torch.sigmoid(bert_embeddings)

        # Create news content representation by concatenating BERT, category, subcategory and entities
        cat_embeddings = self.cat_embeddings(categories)
        news_embeddings = self.news_embeddings(items)
        sub_cat_embeddings = self.sub_cat_embeddings(subcategories)
        entity_embeddings_1 = self.entity_embeddings(entities[:,0])
        news_embedding = torch.cat((news_embeddings, bert_embeddings, cat_embeddings, sub_cat_embeddings, entity_embeddings_1), 1)
        news_embedding = self.news_content_transform(news_embedding)
        news_embedding = torch.sigmoid(news_embedding)

        return news_embedding

    def forward(self, users, items, categories, subcategories, entities):
        user_embeddings = self.get_user_embeddings(users)
        news_embeddings = self.get_news_embeddings(items, categories, subcategories, entities)
        dot_prod = torch.sum(torch.mul(user_embeddings, news_embeddings), 1)
        return torch.sigmoid(dot_prod)


def train_epoch(model, sample_data, epoch, optimizer, criterion):
    total_loss = 0
    for batch_num, batch in enumerate(sample_data):

        # get the inputs - data is a user_id and news_id which looks up the embedding, and a label
        user_ids, news_ids, category_ids, subcategory_ids, entities, labels = batch

        # zero to parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        prediction = model(user_ids, news_ids, category_ids, subcategory_ids, entities)
        loss = criterion(prediction.view(-1), labels)
        loss.backward()
        optimizer.step()

        # keep track of statistics
        total_loss += loss

    print("Total loss after epoch {}: {} ({} avg)".format(epoch+1, total_loss, total_loss / len(sample_data)))


def train_model(model, data_loader):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=l2_regularization)
    for epoch in range(epochs):

        # create data for this epoch
        sample_data = data_loader.sample_training_data(batch_size, negative_sample_size)

        # train
        train_epoch(model, sample_data, epoch, optimizer, criterion)

        # evaluate every epoch
        eval_model(model, data_loader, train=True)
        eval_model(model, data_loader, 1.0)

        # save embeddings if better auc - TODO
        save_embeddings(model, data_loader)


def eval_model(model, data_loader, sample_prob=1.0, train=False):
    sample_data = data_loader.sample_valid_data(sample_prob, train=train)
    with torch.no_grad():

        all_predictions = []
        all_labels = []

        for impression in sample_data:  # make an iterator instead
            user_ids, news_ids, category_ids, subcategory_ids, entities, labels = impression
            prediction = model(user_ids, news_ids, category_ids, subcategory_ids, entities).view(-1)

            all_predictions.append(prediction.detach().numpy())
            all_labels.append(labels.detach().numpy())

        metrics = {
            "auc": group_auc(all_predictions, all_labels),
            "mrr": mrr(all_predictions, all_labels),
            "ndcg@5": ndcg(all_predictions, all_labels, 5),
            "ndcg@10": ndcg(all_predictions, all_labels, 10)
        }
        print(metrics)


def save_embeddings(model, data_loader):
    user_map = data_loader.users()
    news_map = data_loader.news()
    users = torch.LongTensor(range(len(user_map)))
    news, cats, subcats, entities = data_loader.get_news_content_tensors()
    user_embeddings = model.get_user_embeddings(users)
    news_embeddings = model.get_news_embeddings(news, cats, subcats, entities)
    write_embeddings(user_map, user_embeddings, "user_embeddings.tsv")
    write_embeddings(news_map, news_embeddings, "news_embeddings.tsv")


def write_embeddings(id_to_index_map, embeddings, file_name):
    with open(os.path.join(data_dir, file_name), "w") as f:
        for id, index in id_to_index_map.items():
            f.write("{}\t{}\n".format(id, ",".join(["%.6f" % i for i in embeddings[index].tolist()])))


def read_bert_embeddings(data_loader, train_embeddings_file, valid_embeddings_file):
    embeddings = np.zeros([len(data_loader.news()), 512])
    read_bert_embeddings_from_file(data_loader, train_embeddings_file, embeddings)
    read_bert_embeddings_from_file(data_loader, valid_embeddings_file, embeddings)
    return torch.FloatTensor(embeddings)


def read_bert_embeddings_from_file(data_loader, embeddings_file, embeddings):
    if not os.path.exists(embeddings_file):
        print("{} not found.".format(embeddings_file))
        sys.exit(1)
    print("Reading data from " + embeddings_file)

    with io.open(embeddings_file, "r", encoding="utf-8") as f:
        for line in f:
            embeddings_data = line.split("\t")
            news_id = embeddings_data[0]
            if news_id not in data_loader.news():
                continue # not used
            title_and_abstract_embedding = [ float(i) for i in embeddings_data[1].split(",") ]
            news_index = data_loader.news()[news_id]
            embeddings[news_index] = title_and_abstract_embedding


def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def main():
    set_random_seed(1)

    # read data
    data_loader = MindData(train_news_file, train_impressions_file, valid_news_file, valid_impressions_file)
    num_users = len(data_loader.users())
    num_news = len(data_loader.news())
    num_categories = len(data_loader.categories())
    num_subcategories = len(data_loader.subcategories())
    num_entities = len(data_loader.entities())

    # read BERT embeddings
    bert_embeddings = read_bert_embeddings(data_loader, train_embeddings_file, valid_embeddings_file)

    # create model
    model = ContentBasedModel(num_users, num_news, num_categories, num_subcategories, num_entities, embedding_size, bert_embeddings)

    # train model
    train_model(model, data_loader)


if __name__ == "__main__":
    main()
