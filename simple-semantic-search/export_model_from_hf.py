# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from transformers import BertModel
import torch

encoder = BertModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Vespa bert embedder expects these inputs and outputs
# Vespa implements the pooling, default average

input_names = ["input_ids", "attention_mask", "token_type_ids"]
output_names = ["output_0"]

input_ids = torch.ones(1,32, dtype=torch.int64)
attention_mask = torch.ones(1,32,dtype=torch.int64)
token_type_ids = torch.zeros(1,32,dtype=torch.int64)
args = (input_ids, attention_mask, token_type_ids)
torch.onnx.export(encoder,
  args=args,
  f="model/minilm-l6-v2.onnx",
  do_constant_folding=True,
  input_names = input_names,
  output_names = output_names,
  dynamic_axes = {
    "input_ids": {0: "batch", 1:"batch"},
    "attention_mask": {0: "batch", 1: "batch"},
    "token_type_ids": {0: "batch", 1: "batch"},
    "output_0": {0: "batch"},
  },
  opset_version=14)
