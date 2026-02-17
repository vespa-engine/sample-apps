# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

import sys
import json
import numpy as np
import numpy.random as r
import mmh3

file = sys.argv[1]
vectors = np.load(file)

for index in range(0, vectors.shape[0]):
    if 0 == r.randint(0, 8):
        vector = vectors[index].astype(np.float32)
        id = mmh3.hash(vector.tobytes())  # 32 bits signed int
        doc = {
            "put": "id:laion:centroid::%i" % id,
            "fields": {"id": id, "vector": {"values": vector.tolist()}},
        }
        print(json.dumps(doc))
