#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

# An example of training an LTR model using LightGBM

import lightgbm as lgb
import numpy
import pandas as pd
import json

def rr(a,n=100):
  a = a[0:n]
  n = sum(a)
  if n == 0:
    return 0.0
  else:
    return 1/(1 + (numpy.argmax(a)))

def calculate_mrr_for_column(df, query_id='query_id', column='gbdt'):
  mrr = df.groupby(query_id).apply(
    lambda d: rr(d.sort_values(by=[column],ascending=False)['relevant'].values,n=100)
  ).mean()
  return mrr

def load_features(file):
  df = pd.read_csv(file)
  query_groups=df['qid'].value_counts().sort_index().values
  target = df['relevant'].values
  #We drop two features which are not used in this experiment + rest
  t = df.drop(['qid', 'pid', 'relevant', 'closeness(field,text_embedding)', 'nativeProximity(text)',], axis=1)
  #We use t.values here. lgb supports pandas but does not like feature names with , 
  return lgb.Dataset(t.values, target, group=query_groups) 

train = load_features('doc-train-features.csv')
valid = load_features('doc-dev-features.csv')

dev_eval = pd.read_csv("doc-dev-features.csv")
dev_pred = dev_eval['relevant']
qid = dev_eval['qid']
pid = dev_eval['pid']
test = dev_eval.drop(['qid', 'pid', 'relevant', 'closeness(field,text_embedding)', 'nativeProximity(text)',], axis=1)

params = {
  'objective': 'lambdarank',
  'metric': 'ndcg',
  'eval_at': '5,10', 
  'label_gain': [0,1],
  'lambdarank_truncation_level': 10,
  'eta':0.05,
  'num_leaves': 128,
  'min_data_in_leaf': 100,
  'feature_fraction':0.8 
  }

model = lgb.train(params, train , num_boost_round=1000, valid_sets=[train,valid], early_stopping_rounds=50)
pred = model.predict(test.values, num_iteration=model.best_iteration)
rank_evaluation = pd.DataFrame({
  "query_id": qid,
  "pid": pid,
  "gbdt": pred,
  "relevant": dev_pred
})
mrr = calculate_mrr_for_column(rank_evaluation,column='gbdt')
print("Documents (re)-ranked by model {} MRR@100 = {}".format('gbdt',mrr))

model_serialized = model.dump_model(num_iteration=model.best_iteration)
model_json = json.dumps(model_serialized)
columns = test.columns.values.tolist()
for i in range(0,len(columns)):
    name = "\"Column_{}\"".format(i)
    model_json = model_json.replace(name,"\"" + columns[i] + "\"")

with open("docranker.json","w") as fp:
    fp.write(model_json)
