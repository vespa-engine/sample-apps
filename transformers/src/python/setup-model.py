#!/usr/bin/env python3

import os
import sys
import onnx
import transformers
import transformers.convert_graph_to_onnx as onnx_convert

from pathlib import Path

output_file = sys.argv[1]
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
model = transformers.BertForSequenceClassification.from_pretrained(model_name)
pipeline = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

onnx_convert.convert_pytorch(pipeline, opset=11, output=Path(output_file), use_external_format=False)
