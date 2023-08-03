import sys
import ir_datasets

for line in sys.stdin:
    dataset = ir_datasets.create_dataset(queries_tsv="datasets/nfcorpus/test-queries")
    _, query_id, score = line.split()
    if query_id == "all":
        continue
    print(f"{query_id}\t{score}\t{dataset.queries.lookup(query_id).text}")
