<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Hypencoder: query-dependent neural ranking

This sample application reproduces [Hypencoder](https://arxiv.org/abs/2502.05364) (SIGIR '25) in a Vespa rank profile. Hypencoder replaces cosine similarity with a hypernetwork: the query encoder generates the weights of a small query-specific neural network at query time, and that q-net is applied to each document's stored embedding to produce the relevance score.

The app demonstrates running Hypencoder entirely inside Vespa with no custom code: a [hugging-face-embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder) for the passage encoder, an [onnx-model](https://docs.vespa.ai/en/onnx.html) for the query encoder, and a rank profile that expresses the q-net forward pass as tensor expressions.

A forthcoming post on the [Vespa blog](https://blog.vespa.ai/) covers the design and performance characteristics in detail.

Requires at least Vespa 8.338.38.

## Prerequisites

- Docker or Podman.
- [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html) (`brew install vespa-cli` on macOS).
- Python 3.10+ with `pip install -r requirements.txt`.
- Hypencoder's reference implementation (required by `model2onnx.py`):

  ```bash
  git clone https://github.com/jfkback/hypencoder-paper.git
  git -C hypencoder-paper checkout 951ee82ddf2f
  pip install -e ./hypencoder-paper
  ```

## Export the ONNX models

```bash
python model2onnx.py --checkpoint jfkback/hypencoder.2_layer
```

This writes `passage_encoder.onnx`, `query_encoder.onnx`, and `tokenizer.json` to `app/models/`, where `app/services.xml` and `app/schemas/doc.sd` reference them.

For roughly 2-3x faster query-encoder inference on CPU at the cost of some retrieval-quality drift, add `--quantize-int8`.

## Start Vespa

```bash
docker run --detach --name vespa --hostname vespa \
  --publish 8080:8080 --publish 19071:19071 \
  --memory 12g \
  vespaengine/vespa:latest

vespa config set target local
vespa status deploy --wait 300
```

## Deploy

```bash
vespa deploy app --wait 300
```

## Feed

```bash
vespa feed dataset/sample.json
```

The `passage_embedder` runs server-side, producing the 768-d CLS embedding for each document.

## Query

The app exposes four rank profiles:

| profile | what it does |
|---|---|
| `cosine_baseline` | Plain cosine similarity. A reference point for the cost of bi-encoder scoring on this corpus. |
| `hypencoder_onnx` | Full Hypencoder rank-all: the q-net scores every matched document. |
| `hypencoder_rerank` | Cosine first-phase, Hypencoder q-net on the top 100. |
| `hypencoder_lexical_rerank` | BM25 first-phase, Hypencoder q-net on the top 100. |

`encode_query.py` produces the query JSON for each profile. The flag selects the profile:

```bash
# cosine_baseline
python encode_query.py --cosine "tallest mountain in the world" > /tmp/q.json
vespa query --file /tmp/q.json

# hypencoder_onnx (default)
python encode_query.py "tallest mountain in the world" > /tmp/q.json
vespa query --file /tmp/q.json

# hypencoder_rerank
python encode_query.py --rerank "tallest mountain in the world" > /tmp/q.json
vespa query --file /tmp/q.json

# hypencoder_lexical_rerank
python encode_query.py --lexical "tallest mountain in the world" > /tmp/q.json
vespa query --file /tmp/q.json
```

## References

- [Hypencoder paper (arXiv 2502.05364)](https://arxiv.org/abs/2502.05364)
- [jfkback/hypencoder-paper](https://github.com/jfkback/hypencoder-paper) - the original Hypencoder repo
- [Vespa documentation: ONNX models in rank profiles](https://docs.vespa.ai/en/onnx.html)
- [Vespa documentation: hugging-face-embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder)
