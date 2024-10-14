from datasets import load_dataset
import json
import hashlib

hash_object = hashlib.sha256()

ds = load_dataset("PolyAI/banking77", split="train")
labels = dict()
with open("labels-map.txt", "r") as f:
  for line in f:
    id, label_text = line.strip().split("\t")
    labels[int(id)] = label_text.strip()

for row in ds:
  text = row['text'].replace('\n', '')
  label = int(row['label'])
  hash_object.update((text + str(label)).encode())
  vespa_doc = {
    "put": "id:banking77:train::" + hash_object.hexdigest(),
    "fields": {
      "text": text,
      "label": labels[label]
    }
  }
  print(json.dumps(vespa_doc))

