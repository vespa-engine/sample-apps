#!/usr/bin/env python3
"""Export lightonai/ColBERT-Zero to ONNX with INT8 quantisation.

Uses pylate-onnx-export to produce an optimised ONNX model and copies
the artefacts into the Vespa application package at app/models/.

Install the export dependency first:
    uv pip install pylate-onnx-export

Usage:
    uv run python export_model.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

MODEL_NAME = "lightonai/ColBERT-Zero"
APP_MODELS_DIR = Path(__file__).resolve().parent / "src" / "main" / "application" / "models"


def main() -> None:
    try:
        from colbert_export.export import export_model
    except ImportError:
        sys.exit(
            "pylate-onnx-export is not installed.\n"
            "Install it with:  uv pip install pylate-onnx-export"
        )

    print(f"Exporting {MODEL_NAME} to ONNX (FP32 + INT8) ...")
    output_path = export_model(
        model_name=MODEL_NAME,
        output_dir=Path("models") / "ColBERT-Zero",
        quantize=True,
        verbose=True,
        force=True,
    )

    # Copy artefacts into the Vespa application package
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    int8_model = output_path / "model_int8.onnx"
    if not int8_model.exists():
        sys.exit(f"Expected INT8 model not found at {int8_model}")

    # Find tokenizer.json – pylate-onnx-export places it alongside the model
    tokenizer = output_path / "tokenizer.json"
    if not tokenizer.exists():
        # Fall back: download directly from HuggingFace
        print("tokenizer.json not in export dir – downloading from HuggingFace ...")
        try:
            from huggingface_hub import hf_hub_download

            tokenizer = Path(
                hf_hub_download(repo_id=MODEL_NAME, filename="tokenizer.json")
            )
        except Exception as exc:
            sys.exit(f"Could not locate tokenizer.json: {exc}")

    shutil.copy2(int8_model, APP_MODELS_DIR / "model_int8.onnx")
    shutil.copy2(tokenizer, APP_MODELS_DIR / "tokenizer.json")

    # Write a small metadata file consumed by feed.py
    config = {
        "model_name": MODEL_NAME,
        "embedding_dim": 128,
        "onnx_model": str(APP_MODELS_DIR / "model_int8.onnx"),
        "tokenizer": str(APP_MODELS_DIR / "tokenizer.json"),
    }
    # Also check onnx_config.json from export
    onnx_cfg_path = output_path / "onnx_config.json"
    if onnx_cfg_path.exists():
        with open(onnx_cfg_path) as f:
            onnx_cfg = json.load(f)
        config["embedding_dim"] = onnx_cfg.get("embedding_dim", 128)

    meta_path = APP_MODELS_DIR / "model_config.json"
    with open(meta_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel artefacts copied to {APP_MODELS_DIR}/")
    print(f"  model_int8.onnx  ({int8_model.stat().st_size / 1e6:.1f} MB)")
    print(f"  tokenizer.json")
    print(f"  model_config.json")
    print("\nVespa application is ready to deploy.")


if __name__ == "__main__":
    main()
