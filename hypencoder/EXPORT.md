<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Re-exporting the ONNX models

The [README](README.md) downloads pre-built ONNX exports, which is all you need to run the
sample app. This document covers rebuilding them yourself with `model2onnx.py`, which you
need only if you want to change the export itself, for instance to produce an INT8
quantized query encoder, or to check the published files against your own build.

## Environment

`model2onnx.py` needs PyTorch and the Hypencoder reference implementation. These are
deliberately kept out of `requirements.txt` so they play no part in running the app; use a
throwaway virtualenv:

```bash
python -m venv /tmp/export && source /tmp/export/bin/activate
pip install "torch>=2.2" "transformers==4.48.2" "onnx>=1.16" "onnxruntime>=1.20"

git clone https://github.com/jfkback/hypencoder-paper.git
git -C hypencoder-paper checkout 951ee82ddf2f
pip install -e ./hypencoder-paper
```

`transformers` is held at 4.48.2 deliberately. The Hypencoder reference implementation
calls `AutoModel.from_pretrained()` inside `Hypencoder.__init__`, and `transformers>=5`
wraps `cls(config)` in a meta-device context that rejects the nested load, failing with:

```
RuntimeError: You are using `from_pretrained` with a meta device context manager
or `torch.set_default_device('meta')`.
```

4.48.2 is the version the checkpoints' `config.json` records.

The reference implementation is installed editable from a local clone because it is not on
PyPI, and a plain `pip install git+...` does not work: upstream ships its subpackages as
PEP 420 implicit namespace packages while `setup.py` declares only the top-level package,
so a non-editable install copies an empty `__init__.py` and nothing else.

## Export

```bash
python model2onnx.py --checkpoint jfkback/hypencoder.2_layer
```

This writes `passage_encoder.onnx`, `query_encoder.onnx` and `tokenizer.json` to
`app/models/`.

`app/services.xml` fetches the passage encoder and tokenizer from a URL by default. To use
your locally exported copies instead, change those `url=` attributes back to
`path="models/passage_encoder.onnx"` and `path="models/tokenizer.json"`.

Adding `--quantize-int8` runs ONNX Runtime dynamic INT8 quantization on the query encoder:
smaller file, slightly different scores, and a speed difference that varies by platform.

## Other checkpoint depths

Upstream publishes q-nets with 2, 4, 6 and 8 blocks, and all four are exported in the
same Hugging Face repository as `4_layer/`, `6_layer/` and `8_layer/`, so you do not need
to re-export to try them.

The rank profile in `app/schemas/doc.sd` implements the **2 block** version: it reads `W0`,
`b0`, `W1`, `b1` and `Wout`. A deeper checkpoint emits more `W{i}`/`b{i}` outputs, and
those extra blocks have to be added to the rank expression. Without that, the profile
silently scores using only the first two blocks and the final projection.
