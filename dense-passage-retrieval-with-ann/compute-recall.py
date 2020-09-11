#!/usr/bin/env python3 
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import sys
import numpy as np

with open(sys.argv[1],'r') as f:
  result = json.load(f)
n = len(result)

accuracy_10 = 0
accuracy_20 = 0
accuracy_100 = 0
found = 0
mrr_total = []

for q in result:
  found_answer = False
  position = 1000
  for i,h in enumerate(q['ctxs']):
    if h['has_answer']:
      found_answer = True
      position = i + 1
      break

  if found_answer:
    found = found + 1

  if position <= 10:
    accuracy_10 = accuracy_10 + 1

  if position <= 20:
    accuracy_20 = accuracy_20 + 1

  if position <= 100:
    accuracy_100 = accuracy_100 + 1

  mrr = 1./position if position else 0
  mrr_total.append(mrr)

print("N is %i" % n)
print("Accuracy@100 %f " % (found/n))
print("Recall@100 %f " % (accuracy_100/n))
print("Recall@20 %f " % (accuracy_20/n))
print("Recall@10 %f " % (accuracy_10/n))
print("MRR@100 %f " % (np.mean(mrr_total)))

