#!/usr/bin/env python3 
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import onnx   
import transformers   
import transformers.convert_graph_to_onnx as onnx_convert   
from pathlib import Path 
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base', return_dict=True)
pipeline = transformers.Pipeline(model=model, tokenizer=tokenizer)  
onnx_convert.convert_pytorch(pipeline, opset=11, output=Path("question_encoder.onnx"), use_external_format=False) 
#onnx_convert.quantize(Path("question_encoder.onnx")) 
