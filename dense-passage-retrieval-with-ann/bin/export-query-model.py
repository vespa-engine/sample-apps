#!/usr/bin/env python3

# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import os
import sys
import onnx
import transformers
import transformers.convert_graph_to_onnx as onnx_convert

from pathlib import Path

tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
model = transformers.DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base', return_dict=True)
pipeline = transformers.Pipeline(model=model, tokenizer=tokenizer)

output_file = sys.argv[1]
os.makedirs(os.path.dirname(output_file), exist_ok=True)

onnx_convert.convert_pytorch(pipeline, opset=11, output=Path(output_file), use_external_format=False)
onnx_convert.quantize(Path(output_file))
