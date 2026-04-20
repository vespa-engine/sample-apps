import json

with (
    open("ext/corpus.jsonl", "r") as infile, 
    open("ext/corpus_transformed_full.jsonl", "w") as outfile_full,
    open("ext/corpus_transformed_500000.jsonl", "w") as outfile_500000,
    open("ext/corpus_transformed_50000.jsonl", "w") as outfile_50000,
    open("ext/corpus_transformed_1000.jsonl", "w") as outfile_1000,
    ):
    for line in infile:
        doc = json.loads(line)
        doc_id = doc["docid"]
        transformed = {
            "put": f"id:msmarco:passage::{doc_id}",
            "fields": {
                "text": doc["text"],
                "title": doc["title"],
                "id": doc_id,
            },
        }
        outfile_full.write(json.dumps(transformed) + "\n")
        outfile_500000.write(json.dumps(transformed) + "\n")
        outfile_50000.write(json.dumps(transformed) + "\n")
        outfile_1000.write(json.dumps(transformed) + "\n")
