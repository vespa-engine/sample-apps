# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3
import torch

class Ranker(torch.nn.Module):

    def __init__(self):
        super(Ranker, self).__init__()


    def forward(self, query, documents):
        return torch.sum(query*documents, 1)

ranker = Ranker()
input_names = ["query", "documents"]
output_names = ["score"]

query = torch.ones(1,768, dtype=torch.float)
documents = torch.ones(1,768,dtype=torch.float)
args = (query, documents)
torch.onnx.export(ranker,
                  args=args,
                  f="vespa_innerproduct_ranker.onnx",
                  input_names = input_names,
                  output_names = output_names,
                  dynamic_axes = {
                      "documents": {0: "batch"}
                  },
                  opset_version=15)
