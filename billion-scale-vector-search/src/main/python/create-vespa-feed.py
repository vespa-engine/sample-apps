# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import struct
import numpy as np
import sys
import binascii
import json
import numpy.random as r

graph_vectors = open("graph-vectors.jsonl","w")
if_vectors = open("if-vectors.jsonl","w")

n = 0
if len(sys.argv) == 3:
  n = int(sys.argv[2])

with open(sys.argv[1],'rb') as fp:
  vec_count = struct.unpack('i', fp.read(4))[0]
  vec_dimension = struct.unpack('i', fp.read(4))[0]
  if n > 0:
    vec_count = n
  for i in range(0, vec_count):
    vector = fp.read(vec_dimension)
    vector = np.frombuffer(vector, dtype=np.int8)
    graph = False
    if r.randint(0,5) == 0:
      graph = True   

    str_vector = str(binascii.hexlify(vector),'utf-8')
    put = {
      "id": "id:spann:vector::%i" % i,
      "fields": {
        "id": i,
        "in_graph": graph,
        "vector": {
          "values": str_vector
        }
      }
    }
    if graph:
      graph_vectors.write(json.dumps(put))
      graph_vectors.write('\n')
    else:
      if_vectors.write(json.dumps(put))
      if_vectors.write('\n')

graph_vectors.close()
if_vectors.close()
