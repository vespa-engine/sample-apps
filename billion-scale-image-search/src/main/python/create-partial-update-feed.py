# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3
from datasets import load_dataset
import json

dataset = load_dataset("ChristophSchuhmann/improved_aesthetics_6plus")

for row in dataset['train']:
  hash = row['hash']
  punsafe = int(100*row['punsafe'])
  aesthetic = int(10*row['AESTHETIC_SCORE'])
  update = {
    "update": "id:laion:image::%i" % hash,
      "fields": {
        "punsafe": {
          "assign": punsafe
        },
        "aesthetic": {
          "assign": aesthetic 
        }
      }
  }
  print(json.dumps(update))
