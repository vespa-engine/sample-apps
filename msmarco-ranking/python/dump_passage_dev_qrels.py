#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import ir_datasets

dataset = ir_datasets.load('msmarco-passage/dev/small')
with open('qrels.dev.small.tsv', 'w') as fp:
    for query_id,doc_id,relevance,iteration in dataset.qrels_iter():
        fp.write("{}\t0\t{}\t1\n".format(query_id,doc_id))
