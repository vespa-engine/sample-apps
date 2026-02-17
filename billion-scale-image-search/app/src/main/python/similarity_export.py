# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

import torch

class Similarity(torch.nn.Module):

  def __init__(self):
    super(Similarity, self).__init__()


  def forward(self, documents):
    return torch.matmul(documents, documents.t())


ranker = Similarity()
input_names = ["documents"]
output_names = ["similarity"]

documents = torch.ones(1,768,dtype=torch.float)
args = (documents)
torch.onnx.export(ranker,
                  args=args,
                  f="vespa_pairwise_similarity.onnx",
                  input_names = input_names,
                  output_names = output_names,
                  dynamic_axes = {
                      "documents": {0: "batch"}
                  },
                  opset_version=15)
