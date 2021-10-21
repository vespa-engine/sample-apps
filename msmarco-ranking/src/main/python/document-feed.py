#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import sys

for line in sys.stdin:
  doc = json.loads(line)
  id = doc['doc_id']
  text = doc['body']
  title = doc['title']
  url = doc['url']
  doc = {
    "put":"id:msmarco:doc::%s" % id,
      "fields": {
        "text": text,
        "id": id,
        "title": title,
        "url": url,
    }
  }
  json.dump(doc,sys.stdout)
  sys.stdout.write('\n')
