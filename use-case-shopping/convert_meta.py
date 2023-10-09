#!/usr/bin/env python3

# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import time
import random
import json

def process(data):
    fields = {}
    fields["asin"] = data["asin"]
    fields["timestamp"] = int(time.time()) - random.randint(0, 60*60*24*365)  # no date in data, set to a random value up to one year back
    fields["title"] = data["title"]
    fields["description"] = data["description"]
    fields["price"] = data["price"]
    fields["rating_stars"] = 0
    fields["rating_count"] = 0
    fields["images"] = [data["imUrl"]]

    if "brand" in data:
        fields["brand"] = data["brand"]

    # In the data set, categories are of the form
    #   [ ["Sports & Outdoors", "Accessories", "Sport Watches"] ]
    # For filtering on categories, these should be matched exactly, so we transform to
    #   [ "Sports & Outdoors", "Sports & Outdoors|Accessories", "Sports & Outdoors|Accessories|Sport Watches"]
    # because there are multiple subcategories with the same name, and
    # we want to maintain the category hierarchy for grouping.
    # For free text search however, we want to match on stemmed terms.
    # We have another field for this, and reverse the categories for better relevance:
    #   "Sport Watches Accessories Sports & Outdoors"
    if "categories" in data:
        fields["categories"] = []
        fields["categories_text"] = ""

        for category in data["categories"]:
            for level in range(len(category)):
                fields["categories"].append("|".join(category[0:level+1]))
            fields["categories_text"] += " ".join(category[::-1])

    if "related" in data:
        related = []
        for key in data["related"]:
            related.extend(data["related"][key])
        fields["related"] = related

    document = {}
    document["put"] = "id:item:item::" + fields["asin"]
    document["fields"] = fields
    return document

def main():
    output_data = []
    lines = 0
    skipped = 0
    for line in sys.stdin.readlines():
        try:
            lines += 1
            if lines % 1000 == 0:
                sys.stderr.write("Processed %d lines so far...\n" % lines)
            processed = process(eval(line))
            if processed is not None:
                output_data.append(processed)
        except Exception as e:
            skipped += 1  # silently skip errors for now
    print(json.dumps(output_data, indent=2))
    sys.stderr.write("Done. Processed %d lines. Skipped %d lines, probably due to missing data.\n" % (lines, skipped))


if __name__ == "__main__":
    main()

