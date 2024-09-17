import csv
import json

csv_file_path = '/Users/radu/gits/sample-apps/examples/training-artifacts/101/ch2/ecommerce/ext/myntra_products_catalog.csv'
jsonl_file_path = '/Users/radu/gits/sample-apps/examples/training-artifacts/101/ch2/ecommerce/ext/products.jsonl'

def csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, mode='r', encoding='utf-8') as csvfile, open(jsonl_file, mode='w', encoding='utf-8') as jsonlfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # strip leading and trailing whitespaces from all values
            row = {k: v.strip() for k, v in row.items()}

            # Create a Vespa write command
            write_command = {
                "put": "id:ecommerce:product::" + row['ProductID'],
                "fields": row
            }
            jsonlfile.write(json.dumps(write_command) + '\n')

csv_to_jsonl(csv_file_path, jsonl_file_path)