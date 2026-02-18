# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import struct
import numpy as np
import json
import os
import sys
import requests
import argparse
from tqdm import tqdm

def get_result(query_vector):

  request_body = {
    'hits': args.hits,
    'spann.clusters': args.clusters,
    'spann.rank-count': args.rank_count,
    'spann.distance-prune-threshold': args.distance_prune_threshold,
    'input.query(q)': query_vector.tolist(),
    'timeout': '10s'
  }

  response = session.post(args.endpoint, json=request_body,timeout=15.0)
  if not response.ok:
    print("Failed request for query {}, response {}, response json {}".format(query_vector, response,response.json()))
    return 0, 0, []

  result = response.json()
  time = result['timing']['searchtime']*1000
  total_hits = result['root']['fields']['totalCount']
  docs = []
  for h in result['root']['children']:
    fields = h.get('fields',None)
    docs.append(fields.get('id'))
  return time,total_hits,docs
 
def read_queries():
  with open(args.query_file, 'rb') as fq:
    q_count = struct.unpack('i', fq.read(4))[0]
    q_dimension = struct.unpack('i', fq.read(4))[0]
    queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))
    return queries

def knn_result_read():
  n, d = map(int, np.fromfile(args.query_gt_file, dtype="uint32", count=2))
  assert os.stat(args.query_gt_file).st_size == 8 + n * d * (4 + 4)
  f = open(args.query_gt_file, "rb")
  f.seek(4+4)
  I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
  D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
  return I, D

def compute_recall_with_distance_ties(true_ids, true_dists, run_ids, count):
  # From https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/plotting/metrics.py#L15
  # This function assumes "true_dists" is monotonic either increasing or decreasing
  found_tie = False
  gt_size = np.shape(true_dists)[0]
  if gt_size==count:
  # nothing fancy to do in this case
    recall =  len(set(true_ids[:count]) & set(run_ids))
  else:
    dist_tie_check = true_dists[count-1] # tie check anchored at count-1 in GT dists
    set_end = gt_size
    for i in range(count, gt_size):
      is_close = abs(dist_tie_check - true_dists[i] ) < 1e-6 
      if not is_close:
        set_end = i
        break

    found_tie = set_end > count
    recall =  len(set(true_ids[:set_end]) & set(run_ids))
 
  return recall, found_tie


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True) 
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--query_gt_file", type=str, required=True)
    parser.add_argument("--hits", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--clusters_additional", type=int, default=6)
    parser.add_argument("--rank_count", type=int, default=1000)
    parser.add_argument("--distance_prune_threshold", type=float, default=0.80)
    parser.add_argument("--certificate", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--queries", type=int, default=0)

    global args
    args = parser.parse_args()

    queries = read_queries()
    truths,distances =  knn_result_read() 
    recalls = []
    times = []
    hits = []

    global session
    session = requests.Session()
    if args.certificate and args.key:
      session.cert = (args.certificate, args.key)
    if args.queries != 0:
        queries = queries[0:args.queries]
    for i in tqdm(range(0,len(queries))):
      query = queries[i]
      time,total_hits,result = get_result(query)
      times.append(time)
      hits.append(total_hits)
      recall_10,found_tie =  compute_recall_with_distance_ties(truths[i], distances[i], result, 10) 
      recalls.append(10*recall_10)
    print("{},{}".format(np.average(recalls), np.average(times)))
      

if __name__ == "__main__":
    main()

