#!/usr/bin/env python3
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
# quantize model

import sys
from pathlib import Path 
from transformers.convert_graph_to_onnx import quantize

input_file = sys.argv[1]
print("Performing quantization of model '{}'".format(input_file))
quantized_model_path =  quantize(Path(input_file))
print("Rename quantized model '{}' to '{}'".format(quantized_model_path.name, input_file))
quantized_model_path.replace(input_file)
