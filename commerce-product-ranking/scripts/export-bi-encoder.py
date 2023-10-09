# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import transformers.convert_graph_to_onnx as onnx_convert
import transformers
import torch 
from pathlib import Path
import torch.nn as nn
import sys

model = sys.argv[1]
model_name = "bergum/" + model
output_file = model + ".onnx"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.eval()
pipeline = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer)
print("Ignore warnings about model not recognized")
onnx_convert.convert_pytorch(pipeline, opset=12, output=Path(output_file), use_external_format=False)
