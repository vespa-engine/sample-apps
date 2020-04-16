import re
import sys
import json

DATA_FILE = sys.argv[1]

def nor(s):
  return re.sub(r' +', " ", re.sub(r'[0-9!"#$%&()*+,./:;<=>?@[\]^_`{|}~\'—–-]+', "", s or "")).lower().strip()

duplicates = 0
docs = {}
with open(DATA_FILE, 'r') as f:
  for doc in json.load(f):
    title = nor(doc['title'])
    if title in docs.keys():
      #print("Duplicate title %s" % doc['title'], file=sys.stderr)
      docs[title].append(doc)
      duplicates = duplicates + 1
    else:
      docs[title] = [doc]
    doc['cited_by'] = []
print("Total duplicates %d" % duplicates, file=sys.stderr)

inbound = 0
for ds in docs.values():
  for doc in ds:
    for ref in doc['bib_entries']:
      title = nor(ref['title'])
      if title in docs.keys():
        for cited in docs[title]:
          cited['cited_by'].append(doc['id'])
        inbound = inbound + 1

print("Total inbound hits %d" % inbound, file=sys.stderr)

to_print = []
for ds in docs.values(): to_print.extend(ds)

print(json.dumps(to_print))

