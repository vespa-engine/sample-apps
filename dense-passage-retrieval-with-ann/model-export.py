#!/usr/bin/env python3 
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import onnx   
import transformers   
import transformers.convert_graph_to_onnx as onnx_convert   
from pathlib import Path 
from transformers import DPRReader, DPRReaderTokenizer 
tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base') 
model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)  
pipeline = transformers.Pipeline(model=model, tokenizer=tokenizer)  
onnx_convert.convert_pytorch(pipeline, opset=11, output=Path("reader.onnx"), use_external_format=False) 
onnx_convert.quantize(Path("reader.onnx")) 
