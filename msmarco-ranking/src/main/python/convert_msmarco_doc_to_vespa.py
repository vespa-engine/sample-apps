# Modified to output Vespa json format, original version
# https://github.com/castorini/docTTTTTquery/blob/master/convert_msmarco_passages_doc_to_anserini.py

import json
import os
import argparse
import gzip 
from tqdm import tqdm

def generate_output_dict(doc, predicted_queries):
    doc_id = doc[0]
    preds = []
    for s in predicted_queries:
      s = s.strip().split("#")
      for k in s:
        preds.append(k)

    update = {
      "update": "id:msmarco:doc::{}".format(doc_id),
      "fields": {
        "doc_t5_query": {
         "assign": preds
        }
      }
    }
    return update

parser = argparse.ArgumentParser(
    description='Concatenate MS MARCO original docs with predicted queries')
parser.add_argument('--original_docs_path', required=True, help='MS MARCO .tsv corpus file.')
parser.add_argument('--doc_ids_path', required=True, help='File mapping segments to doc ids.')
parser.add_argument('--predictions_path', required=True, help='File containing predicted queries.')
parser.add_argument('--output_docs_path', required=True, help='Output file in the Vespa jsonl format.')

args = parser.parse_args()

f_corpus = gzip.open(args.original_docs_path, mode='rt')
f_out = open(args.output_docs_path, 'w')

print('Appending predictions...')
doc_id = None
for doc_id_ref, predicted_queries_partial in tqdm(zip(open(args.doc_ids_path),
                                                      open(args.predictions_path))):
    doc_id_ref = doc_id_ref.strip()
    if doc_id_ref != doc_id:
        if doc_id is not None:
            output_dict = generate_output_dict(doc, predicted_queries)
            f_out.write(json.dumps(output_dict) + '\n')

        doc = next(f_corpus).split('\t')
        doc_id = doc[0]
        predicted_queries = []

    predicted_queries.append(predicted_queries_partial)

output_dict = generate_output_dict(doc, predicted_queries)
f_out.write(json.dumps(output_dict) + '\n')

f_corpus.close()
f_out.close()
print('Done!')

