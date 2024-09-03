#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import torch
from transformers import AutoModel, BertPreTrainedModel


class VespaColBERT(BertPreTrainedModel):
    """ A wrapper around the Hugging Face Transformers model to export to ONNX
    for inference with Vespa colbert-embedder.
    https://docs.vespa.ai/en/embedding.html#colbert-embedder
    """
    def __init__(self, config, dim):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.linear = torch.nn.Linear(config.hidden_size, dim, bias=False)
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        # Projection layer 
        Q = self.linear(Q)
        # Normalization so that each vector is unit length 
        return torch.nn.functional.normalize(Q, p=2, dim=2)


# Replace with your model trained with ColBERT

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--hf_model", type=str, default='answerdotai/answerai-colbert-small-v1') 
parser.add_argument("--dims", type=int, default=96)
parser.add_argument("--out_file", type=str, default="model.onnx")
args = parser.parse_args()

print("Vespa ONNX export for model " + args.hf_model + " with dims " + str(args.dims) + " to " + args.out_file)

vespa_colbert = VespaColBERT.from_pretrained(args.hf_model, dim=args.dims)
out_file_name: str = args.out_file

# These are the default input and output names expected
# by the Vespa colbert-embedder 

input_names = ["input_ids", "attention_mask"]
output_names = ["contextual"]

input_ids = torch.ones(1, 32, dtype=torch.int64)
attention_mask = torch.ones(1, 32, dtype=torch.int64)
args = (input_ids, attention_mask)
torch.onnx.export(
    vespa_colbert,
    args=args,
    f=str(out_file_name),
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "batch"},
        "attention_mask": {0: "batch", 1: "batch"},
        "contextual": {0: "batch", 1: "batch"},
    },
    opset_version=17,
)
print("Vespa ONNX export complete! See https://docs.vespa.ai/en/embedding.html#colbert-embedder for usage in Vespa.ai")
