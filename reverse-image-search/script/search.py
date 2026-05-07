"""Run a reverse-image-search query against a deployed Vespa application.

Loads the same DINOv2 model used by the feeder, embeds the input image, packs
the binary version, and runs a `nearestNeighbor(embedding_binary, q_bin)`
query with the `closeness` rank profile — same path the demo's backend uses.

Environment
-----------
    RIS_VESPA_URL        Vespa endpoint (e.g. http://localhost:8080 for local
                         Docker, or the mTLS URL printed by `vespa deploy` for
                         Vespa Cloud). Required.
    RIS_VESPA_CERT_PATH  Optional. Path to data-plane public cert (Vespa Cloud).
    RIS_VESPA_KEY_PATH   Optional. Path to data-plane private key (Vespa Cloud).
    RIS_MODEL_NAME       Default: facebook/dinov2-base.
    RIS_DEVICE           Default: auto (cuda → mps → cpu).

Usage
-----
    python search.py path/to/query.jpg --hits 10 --ranking hybrid
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from vespa.application import Vespa


_RANKING_PROFILES = {
    "hybrid":         "closeness",                # binary HNSW first-phase + float rerank top 100
    "hybrid_strict":  "closeness_hybrid_strict",  # same, rerank top 1000
    "binary":         "closeness_binary",         # binary HNSW only, no rerank
    "weighted":       "closeness_weighted",       # binary HNSW first-phase + α·float + (1−α)·binary rerank
}


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _embed(image_path: Path, model_name: str, device: str) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    img = Image.open(image_path).convert("RGB")
    with torch.inference_mode():
        batch = processor(images=[img], return_tensors="pt").to(device)
        out = model(**batch)
        cls = out.last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls, dim=-1)
    return cls.detach().to("cpu", dtype=torch.float32).numpy()[0]


def _pack_binary(emb: np.ndarray) -> np.ndarray:
    """Sign-bit pack 768 floats into 96 int8s — matches the schema's
    `binarize | pack_bits` indexing pipeline so the query tensor is
    bit-aligned with the indexed `embedding_binary` field."""
    bits = (emb > 0).astype(np.uint8)
    return np.packbits(bits).view(np.int8)


def _search(
    app: Vespa,
    embedding: np.ndarray,
    hits: int,
    target_hits: int,
    ranking: str,
    alpha: float | None,
) -> dict:
    profile = _RANKING_PROFILES[ranking]
    binary_q = _pack_binary(embedding)
    body = {
        "yql": (
            f"select * from image where "
            f"{{targetHits:{target_hits}}}nearestNeighbor(embedding_binary, q_bin)"
        ),
        "input.query(q)":     embedding.astype(np.float32).tolist(),
        "input.query(q_bin)": binary_q.astype(int).tolist(),
        "ranking.profile":    profile,
        "hits":               hits,
        "presentation.summary": "id-only",
    }
    if ranking == "weighted" and alpha is not None:
        body["input.query(alpha)"] = float(alpha)
    response = app.query(body=body)
    if not response.is_successful():
        raise RuntimeError(f"Vespa query failed: {response.get_json()}")
    return response.json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, help="Path to the query image")
    p.add_argument("--hits", type=int, default=10)
    p.add_argument("--target-hits", type=int, default=200,
                   help="targetHits for nearestNeighbor — affects HNSW exploration depth")
    p.add_argument("--ranking", choices=list(_RANKING_PROFILES), default="hybrid")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="weight on float closeness in the `weighted` profile (1-α on binary)")
    args = p.parse_args()

    url = os.environ["RIS_VESPA_URL"].rstrip("/")
    cert = os.environ.get("RIS_VESPA_CERT_PATH") or None
    key = os.environ.get("RIS_VESPA_KEY_PATH") or None
    kwargs: dict = {"url": url}
    if cert and key:
        kwargs["cert"] = cert
        kwargs["key"] = key
    app = Vespa(**kwargs)

    model_name = os.environ.get("RIS_MODEL_NAME", "facebook/dinov2-base")
    device = _device(os.environ.get("RIS_DEVICE", "auto"))
    embedding = _embed(args.image, model_name, device)

    payload = _search(app, embedding, args.hits, args.target_hits, args.ranking, args.alpha)
    children = (payload.get("root") or {}).get("children") or []
    for hit in children:
        fields = hit.get("fields") or {}
        mf = fields.get("matchfeatures") or {}
        print(json.dumps({
            "id":               fields.get("id"),
            "filename":         fields.get("filename"),
            "split":            fields.get("split"),
            "label":            fields.get("label"),
            "score":            hit.get("relevance"),
            "binary_closeness": mf.get("closeness(field,embedding_binary)"),
            "float_closeness":  mf.get("closeness_emb"),
        }))


if __name__ == "__main__":
    main()
