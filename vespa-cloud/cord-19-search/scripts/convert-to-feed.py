#!/usr/bin/env python3
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import json

json_file = sys.argv[1]
with open(json_file, 'r') as f:
  data = json.load(f)
  for doc in data:
    vespa_doc = {
      'put': 'id:covid-19:doc::%s' % doc['id'], 
      'fields': doc
    }
    print(json.dumps(vespa_doc))
