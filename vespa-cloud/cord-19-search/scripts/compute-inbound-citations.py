import re
import sys
import json

DATA_FILE = sys.argv[1]

def nor(s):
  if not s: s = "nope"
  return re.sub(r'[0-9!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]+', "", s).lower()

duplicates = 0
docs = {}
with open(DATA_FILE, 'r') as f:
  for doc in json.load(f):
    if nor(doc['title']) in docs.keys():
      #print("Duplicate title %s" % doc['title'], file=sys.stderr)
      duplicates = duplicates + 1
    docs[nor(doc['title'])] = doc
    doc['cited_by'] = []
print("Total duplicates %d" % duplicates, file=sys.stderr)

inbound = 0
for title in docs.keys():
  doc = docs[title]
  for ref in doc['bib_entries']:
    if nor(ref['title']) in docs.keys():
      docs[nor(ref['title'])]['cited_by'].append(doc['id'])
      inbound = inbound + 1
print("Total inbound hits %d" % inbound, file=sys.stderr)

print(json.dumps(list(docs.values())))

