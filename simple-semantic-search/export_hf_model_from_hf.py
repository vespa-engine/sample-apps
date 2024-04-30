#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#
#This script exports an embedding model to be 
#used with https://docs.vespa.ai/en/embedding.html#huggingface-embedder

from transformers import AutoModel, AutoTokenizer
import torch
import argparse
import os

def export_model(model_id, output_dir, quantize=False):
	if not os.path.exists(output_dir):
		print("Output directory '{}' does not exist".format(output_dir))
		return

	embedder = AutoModel.from_pretrained(model_id)
	tokenizer = AutoTokenizer.from_pretrained(model_id)

	input_names = ["input_ids", "attention_mask", "token_type_ids"]
	output_names = ["last_hidden_state"]

	input_ids = torch.ones(1,32, dtype=torch.int64)
	attention_mask = torch.ones(1,32,dtype=torch.int64)
	token_type_ids = torch.zeros(1,32,dtype=torch.int64)
	args = (input_ids, attention_mask, token_type_ids)

	f=os.path.join(output_dir,"model.onnx")
	print("Exporting onnx model to {}".format(f))
	torch.onnx.export(embedder,
  	args=args,
  	f=f,
  	do_constant_folding=True,
  	input_names = input_names,
  	output_names = output_names,
  	dynamic_axes = {
    	"input_ids": {0: "batch_size", 1:"dyn"},
    	"attention_mask": {0: "batch_size", 1:"dyn"},
    	"token_type_ids": {0: "batch_size", 1:"dyn"},
    	"last_hidden_state": {0: "batch_size", 1:"dyn"},
  	},
  	opset_version=14)
	files = tokenizer.save_pretrained(output_dir)
	keep_file = os.path.join(output_dir,"tokenizer.json")
	for i in range(0,len(files)):
		file = files[i]
		if file != keep_file and os.path.exists(file):
			print("Deleting unneeded config file {}".format(file))
			os.remove(file)
	if quantize:
		print("Quantizing model for accelerated CPU-inferencing using dynamic quantization")
		import onnx
		from onnxruntime.quantization import quantize_dynamic
		model_output = f.replace(".onnx","_quantized.onnx")
		print("Quantized model will be saved to {}".format(model_output))
		quantize_dynamic(model_input=f,model_output=model_output)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--hf_model", type=str, required=True) 
	parser.add_argument("--output_dir", type=str, required=True) 
	parser.add_argument("--quantize", action="store_true")
	parser.add_argument("--patch_tokenizer", action="store_true")
	args = parser.parse_args()
	export_model(args.hf_model, args.output_dir, args.quantize)
	if args.patch_tokenizer:
		import json
		print("Patching tokenizer")
		path = os.path.join(args.output_dir,"tokenizer.json")
		with open(path, "r") as f:
			tokenizer = json.load(f)
		model = tokenizer.get('model',None)
		if model and 'byte_fallback' in model:
			del model['byte_fallback']
			with open(path, "w") as f:
				f.write(json.dumps(tokenizer))



if __name__ == "__main__":
	main()
