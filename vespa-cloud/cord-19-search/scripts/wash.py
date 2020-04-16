import re
import sys
import json

def nor(s):
  return re.sub(r' +', " ", re.sub(r'[0-9!"#$%&()*+,./:;<=>?@[\]^_`{|}~\'—–-]+', "", s or "")).lower().strip()

DATA_FILE = sys.argv[1]
BLACKLIST_FILE = sys.argv[2]

blacklist = {}
with open(BLACKLIST_FILE, 'r') as b:
  for line in b:
    blacklist[line.strip()] = None

duplicates = []
docs = {}
with open(DATA_FILE, 'r') as f:
  for doc in json.load(f):
    title = nor(doc['title'])
    if not title in blacklist.keys() and not title == "":
      if title in docs.keys():
        docs[title].append(doc)
        if not title in duplicates:
          duplicates.append(title)
      else:
        docs[title] = [doc]
print("Total duplicates %d" % len(duplicates), file=sys.stderr)
#print("\n".join(duplicates), file=sys.stderr)

to_print = []
for ds in docs.values(): to_print.extend(ds)

print(json.dumps(to_print))

