#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
# Simple util to convert ir_datasets export json format to vespa feed format

import json
import sys

for line in sys.stdin:
  doc = json.loads(line)
  id = doc['doc_id']
  text = doc['text']
  doc = {
    "put":"id:msmarco:passage::%s" % id,
      "fields": {
        "text": text,
        "id": id,
    }
  }
  json.dump(doc,sys.stdout)
  sys.stdout.write('\n')
