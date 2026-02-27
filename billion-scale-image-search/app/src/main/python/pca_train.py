# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

from sklearn.decomposition import IncrementalPCA
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

pca128 = IncrementalPCA(n_components=128)
pca64 = IncrementalPCA(n_components=64)

files = ["laion5b_100m_part_{}_of_10.parquet".format(i) for i in range(1, 11)]
for f in tqdm(files):
  parquet_file = pq.ParquetFile(f)
  for batch in parquet_file.iter_batches(batch_size=50000, columns=["vector"]):
    vectors = np.array(batch.column("vector").to_pylist(), dtype=np.float32)
    pca128 = pca128.partial_fit(vectors)
    pca64  = pca64.partial_fit(vectors)

pca128_comp = np.asarray(pca128.components_)
np.save("pca-128-components.npy", pca128_comp)

pca64_comp = np.asarray(pca64.components_)
np.save("pca-64-components.npy", pca64_comp)
