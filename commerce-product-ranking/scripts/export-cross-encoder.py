# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers.convert_graph_to_onnx as onnx_convert
import transformers
import torch 
from pathlib import Path
import torch.nn as nn

import sys
name = sys.argv[1]

cross_model = "bergum/" + name
output_file = name + ".onnx"
tokenizer = AutoTokenizer.from_pretrained(cross_model)
model = AutoModelForSequenceClassification.from_pretrained(cross_model)
model = model.eval()

pipeline = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer)
onnx_convert.convert_pytorch(pipeline, opset=12, output=Path(output_file), use_external_format=False)

#from onnxruntime.quantization import quantize_dynamic, QuantType
#quantize_dynamic(output_file, "productranker_.onnx") 
