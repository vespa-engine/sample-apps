import json
import itertools

SUBSETS = {
    "full": None,
    "500000": 500000,
    "50000": 50000,
    "5000": 5000,
    "1000": 1000,
}

def transform(doc):
    doc_id = doc["docid"]
    return {
        "put": f"id:msmarco:passage::{doc_id}",
        "fields": {
            "text": doc["text"],
            "id": doc_id,
        },
    }

with open("ext/corpus.jsonl", "r") as infile:
    outfiles = {
        name: open(f"ext/corpus_transformed_{name}.jsonl", "w")
        for name in SUBSETS
    }
    try:
        for i, line in enumerate(infile):
            transformed = transform(json.loads(line))
            serialized = json.dumps(transformed) + "\n"
            for name, limit in SUBSETS.items():
                if limit is None or i < limit:
                    outfiles[name].write(serialized)
    finally:
        for f in outfiles.values():
            f.close()