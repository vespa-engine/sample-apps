# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import numpy as np
import torch
import torch.onnx
import torch.nn as nn


class PairwiseRankerModel(nn.Module):
    def __init__(self, embedding_size):
        super(PairwiseRankerModel, self).__init__()
        self.query_doc_transform = torch.nn.Linear(in_features=embedding_size*2, out_features=embedding_size)
        self.compare_transform = torch.nn.Linear(in_features=embedding_size*2, out_features=1)

    def forward(self, query_embedding, doc_1_embedding, doc_2_embedding):
        query_doc_1_rep = torch.cat((query_embedding, doc_1_embedding), 1)
        query_doc_1_rep = torch.sigmoid(self.query_doc_transform(query_doc_1_rep))
        query_doc_2_rep = torch.cat((query_embedding, doc_2_embedding), 1)
        query_doc_2_rep = torch.sigmoid(self.query_doc_transform(query_doc_2_rep))
        compare = torch.cat((query_doc_1_rep, query_doc_2_rep), 1)
        compare = self.compare_transform(compare)
        return torch.sigmoid(compare)

def main():
    embedding_size = 16
    model = PairwiseRankerModel(embedding_size)

    # Omit training - just export randomly initialized network

    query_data = torch.FloatTensor(np.random.random((1,embedding_size)))
    doc_1_data = torch.FloatTensor(np.random.random((1,embedding_size)))
    doc_2_data = torch.FloatTensor(np.random.random((1,embedding_size)))
    torch.onnx.export(model,
                      (query_data, doc_1_data, doc_2_data),
                      "pairwise_ranker.onnx",
                      input_names = ["query", "doc1", "doc2"],
                      output_names = ["output"],
                      dynamic_axes = {
                          "query":  {0:"batch"},
                          "doc1":   {0:"batch"},
                          "doc2":   {0:"batch"},
                          "output": {0:"batch"},
                      },
                      opset_version=12)


if __name__ == "__main__":
    main()


