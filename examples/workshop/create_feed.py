#!/usr/bin/env python3
import json
import sys

def convert_to_vespa_feed(input_file, output_file):
    """Convert raw JSON to Vespa feed format"""
    with open(input_file, 'r', encoding='utf-8') as f:
        products = json.load(f)
    with open(output_file, 'w', encoding='utf-8') as f:
        for product in products:
            # Create Vespa document structure
            vespa_doc = {
                "put": f"id:products:product::{product['productId']}",
                "fields": {
                    "productId": product["productId"],
                    "sku": product["sku"],
                    "name_index": product["name"],
                    "name_index_bm25": product["name"],
                    "name_index_n_gram": product["name"],
                    "name_attribute": product["name"],
                    "description_index": product["description"],
                    "description_index_bm25": product["description"],
                    "description_index_n_gram": product["description"],
                    "description_attribute": product["description"],
                    "price": product["price"],
                    "pricePerUnit": product["pricePerUnit"],
                    "allergens": [a.strip() for a in product["allergens"].split(",")] if product.get("allergens") else [],
                    "carbonFootprintGram": product["carbonFootprintGram"],
                    "unit": product["unit"],
                    "organic": product["organic"],
                    "gtin": product["gtin"],
                }
            }
            # Write as JSONL (one document per line)
            f.write(json.dumps(vespa_doc, ensure_ascii=False) + '\n')
    print(f"Converted {len(products)} products to Vespa feed format")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_to_vespa_feed(input_file, output_file)
