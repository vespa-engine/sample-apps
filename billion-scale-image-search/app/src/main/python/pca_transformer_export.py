# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

import numpy as np
import torch
import sys

class PCATransformer(torch.nn.Module):

  def __init__(self, weights):
    super(PCATransformer, self).__init__()
    self.weights = weights


  def forward(self, vectors):
    return torch.matmul(weights, vectors)

file = sys.argv[1]


pca_components = np.load(file)
weights = torch.tensor(pca_components,dtype=torch.float32) 

transformer = PCATransformer(weights)
input_names = ["vector"]
output_names = ["reduced_vector"]

vectors = torch.ones(pca_components.shape[1], dtype=torch.float32)
args = (vectors)
torch.onnx.export(transformer,
                  args=args,
                  f="pca_transformer.onnx",
                  input_names = input_names,
                  output_names = output_names,
                  opset_version=12)
