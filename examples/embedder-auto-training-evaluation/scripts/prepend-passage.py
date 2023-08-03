import json
import sys

for line in sys.stdin:
    doc = json.loads(line)

    doc['text'] = f'passage: {doc["text"]}'
    if 'title' in doc:
        doc['title'] = f'passage: {doc["title"]}'

    print(json.dumps(doc))
