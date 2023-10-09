#! /usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import csv
import argparse
import json
import time

parser = argparse.ArgumentParser()

parser.add_argument("--file", "-f", type=str, required=True)
args = parser.parse_args()

pattern = '%Y-%m-%d %H:%M:%S'

with open(args.file, mode='r') as infile:
    reader = csv.DictReader(infile)
    for idx,row in enumerate(reader):
        epoch = int(time.mktime(time.strptime(row['date'], pattern)))
        del row['date']
        row['date'] = epoch
        part = {}
        part['put'] = 'id:purchase:purchase::' + str(idx)
        part['fields'] = row
        print(json.dumps(part, indent=0, sort_keys=True).replace('\n',''))
