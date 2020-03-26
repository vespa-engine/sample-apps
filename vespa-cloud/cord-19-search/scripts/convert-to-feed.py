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
