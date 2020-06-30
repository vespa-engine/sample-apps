#! /usr/bin/env python3

import os
import sys
import json
import torch

from transformers import AutoModelForSequenceClassification
from transformers.pipelines import pipeline


app_dir = sys.argv[1]
model_name = sys.argv[2]
sequence_length = int(sys.argv[3])

tokenizer_name = model_name
framework = "pt"
onnx_opset_version = 11

model_dir = os.path.join(app_dir, "models")
model_output = os.path.join(model_dir, "rankmodel.onnx")
constants_dir = os.path.join(app_dir, "constants")
seq_classification_weights_output = os.path.join(constants_dir, "seq_weights.json")
seq_classification_bias_output = os.path.join(constants_dir, "seq_bias.json")


# The following is adapted from the convert_graph_to_onnx.py
# script in the transformers library.


def load_model():
    print("Loading model (model: {}, tokenizer: {})".format(model_name, tokenizer_name))
    return pipeline("feature-extraction", model=model_name, tokenizer=tokenizer_name, framework=framework)


def flatten(x):
    for i in x:
        if isinstance(i, (tuple, list)):
            for j in flatten(i):
                yield j
        else:
            yield i


def infer_inputs_outputs(nlp):
    tokens = nlp.tokenizer.encode_plus("This is a sample output",
                                       return_tensors=framework,
                                       pad_to_max_length=True,
                                       max_length=sequence_length)
    input_names = list(tokens.keys())
    outputs = list(flatten(nlp.model(**tokens)))
    output_names = ["output_{}".format(i) for i in range(len(outputs))]
    return input_names, output_names, tokens


def export_model(nlp, inputs, outputs, tokens):
    print("Exporting '{}' to application package...".format(model_name))
    model_args_name = nlp.model.forward.__code__.co_varnames
    model_args, ordered_inputs = [], []
    for arg_name in model_args_name[1:]:
        if arg_name in inputs:
            ordered_inputs.append(arg_name)
            model_args.append(tokens[arg_name])
    inputs = ordered_inputs
    model_args = tuple(model_args)

    print("Inputs: " + ", ".join(inputs))
    print("Outputs: " + ", ".join(outputs))

    os.makedirs(model_dir, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
             nlp.model,
             model_args,
             f = model_output,
             input_names = inputs,
             output_names = outputs,
             # dynamic_axes = dynamic_axes,  # we don't use dynamic axes
             do_constant_folding = True,
             use_external_data_format = False,
             enable_onnx_checker = True,
             opset_version=onnx_opset_version,
        )


def to_vespa_tensor_format_2d(tensor, dim_names = ["d0", "d1"]):
    cells = []
    for d0 in range(tensor.shape[0]):
        for d1 in range(tensor.shape[1]):
            cells.append( { "address": { dim_names[0]:"{:d}".format(d0), dim_names[1]:"{:d}".format(d1) }, "value": tensor[d0,d1].astype(float) } )
    return { "cells": cells }


def to_vespa_tensor_format_1d(tensor, dim_names = ["d0"]):
    cells = []
    for d0 in range(tensor.shape[0]):
        cells.append( { "address": { dim_names[0]:"{:d}".format(d0) }, "value": tensor[d0].astype(float) } )
    return { "cells": cells }


# The above does not export the tensors used for sequnce classification.
# We export them manually here:
def export_tensors_for_sequence_classification():
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    print("Exporting tensor for sequence classification to application package...")

    os.makedirs(constants_dir, exist_ok=True)

    weights = model.classifier.weight.detach().numpy()
    print("Shape of weights: {}".format(weights.shape))
    with open(seq_classification_weights_output, "w") as f:
        json.dump(to_vespa_tensor_format_2d(weights, ["d1", "d2"]), f, indent=2)

    bias = model.classifier.bias.detach().numpy()
    print("Shape of bias: {}".format(bias.shape))
    with open(seq_classification_bias_output, "w") as f:
        json.dump(to_vespa_tensor_format_1d(bias, ["d1"]), f, indent=2)


def main():
    nlp = load_model()
    inputs, outputs, tokens = infer_inputs_outputs(nlp)
    export_model(nlp, inputs, outputs, tokens)
    export_tensors_for_sequence_classification()


if __name__ == "__main__":
    main()

