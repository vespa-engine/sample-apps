# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

import torch.nn as nn
import torch

class CustomEmbeddingSimilarity(nn.Module):

	def __init__(self, dimensionality=384):
		super(CustomEmbeddingSimilarity, self).__init__()
		self.fc1 = nn.Linear(2*dimensionality, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 1)

	def forward(self, query , document):
		x = torch.cat((query, document), dim=1)
		x = nn.functional.relu(self.fc1(x))
		x = nn.functional.relu(self.fc2(x))
		x = nn.functional.relu(self.fc3(x))
		return torch.sigmoid(self.fc4(x))

dim = 384
ranker = CustomEmbeddingSimilarity(dimensionality=dim)

# Train the ranker model ..

# Export to ONNX for inference with Vespa 

input_names = ["query","document"]
output_names = ["similarity"]
document = torch.ones(1,dim,dtype=torch.float)
query = torch.ones(1,dim,dtype=torch.float)
args = (query,document)
torch.onnx.export(ranker,
                  args=args,
                  f="custom_similarity.onnx",
                  input_names = input_names,
                  output_names = output_names,
                  opset_version=15)
