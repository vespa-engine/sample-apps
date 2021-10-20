#! /usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import numpy as np
from sklearn.metrics import roc_auc_score


def group_auc(preds, labels):
    group_auc = np.mean(
        [
            roc_auc_score(each_labels, each_preds)
            for each_labels, each_preds in zip(labels, preds)
        ]
    )
    return round(group_auc, 4)


def dcg_score(y_true, y_score, k=10):
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def ndcg(preds, labels, k):
    ndcg_temp = np.mean(
        [
            ndcg_score(each_labels, each_preds, k)
            for each_labels, each_preds in zip(labels, preds)
        ]
    )
    return round(ndcg_temp, 4)


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def mrr(preds, labels):
    mean_mrr = np.mean(
        [
            mrr_score(each_labels, each_preds)
            for each_labels, each_preds in zip(labels, preds)
        ]
    )
    return round(mean_mrr, 4)



