#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

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
