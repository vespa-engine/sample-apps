# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

from sklearn.decomposition import IncrementalPCA
import numpy as np
from tqdm import tqdm

pca128 = IncrementalPCA(n_components=128)
pca64 = IncrementalPCA(n_components=64)

files = ["{:04d}".format(i) for i in range(0,2314)]
sample = np.random.choice(files, size=200)
for s in tqdm(sample):
  vectors = np.load("img_emb_%s.npy" % s)
  pca128 = pca128.partial_fit(vectors)
  pca64  = pca64.partial_fit(vectors)

pca128_comp = np.asarray(pca128.components_)
np.save("pca-128-components.npy", pca128_comp)

pca64_comp = np.asarray(pca64.components_)
np.save("pca-64-components.npy", pca64_comp)
