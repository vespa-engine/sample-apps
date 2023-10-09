#!/usr/bin/env python3
#Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import json
import sys

for line in sys.stdin:
  doc = json.loads(line)
  id = doc['doc_id']
  
  vespa_doc = {
    "put": "id:covid-19:doc::%s" % id,
      "fields": {
        "cord_uid": id, 
        "title": doc.get("title"),
        "abstract": doc["text"]	
      }
  }
  print(json.dumps(vespa_doc))
