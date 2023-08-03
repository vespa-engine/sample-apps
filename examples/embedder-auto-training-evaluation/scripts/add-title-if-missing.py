import json
import sys

for line in sys.stdin:
    doc = json.loads(line)
    doc.setdefault('title', '-')

    print(json.dumps(doc))
