#!/usr/bin/env python3

# Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import time
import random
import json

def process(data):
    fields = {}
    fields["asin"] = data["asin"]
    fields["timestamp"] = data["unixReviewTime"]
    fields["reviewer_id"] = data["reviewerID"]
    fields["reviewer_name"] = data["reviewerName"]
    fields["title"] = data["summary"]
    fields["text"] = data["reviewText"]
    fields["stars"] = int(data["overall"])

    fields["upvotes"] = int(data["helpful"][0])
    fields["downvotes"] = int(data["helpful"][1] - data["helpful"][0])

    document = {}
    document["put"] = "id:review:review::" + fields["asin"] + "-" + fields["reviewer_id"]
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

