# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3
import transformers.convert_graph_to_onnx as onnx_convert
import sys
from pathlib import Path
onnx_convert.quantize(Path(sys.argv[1]))
