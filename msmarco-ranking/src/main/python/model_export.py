#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertConfig
import torch 
import torch.nn as nn
import sys

class VespaColBERT(BertPreTrainedModel):

  def __init__(self,config):
    super().__init__(config)
    self.bert = BertModel(config)
    self.linear = nn.Linear(config.hidden_size, 32, bias=False)
    self.init_weights()

  def forward(self, input_ids, attention_mask):
    Q = self.bert(input_ids,attention_mask=attention_mask)[0]
    Q = self.linear(Q)
    return torch.nn.functional.normalize(Q, p=2, dim=2)  


print("Downloading model from Huggingface")
colbert_query_encoder = VespaColBERT.from_pretrained("vespa-engine/col-minilm")
out_file = sys.argv[1]

print("Exporting model to ONNX format to '{}'".format(out_file))

#Export model to ONNX for serving in Vespa 

input_names = ["input_ids", "attention_mask"]
output_names = ["contextual"]
#input, max 32 query term
input_ids = torch.ones(1,32, dtype=torch.int64)
attention_mask = torch.ones(1,32,dtype=torch.int64)
args = (input_ids, attention_mask)

colbert_query_encoder.eval()
torch.onnx.export(colbert_query_encoder,
  args=args,
  f=out_file,
  input_names = input_names,
  output_names = output_names,
  dynamic_axes = {
    "input_ids": {0: "batch"},
    "attention_mask": {0: "batch"},
    "contextual": {0: "batch"},
  },
  opset_version=11)
