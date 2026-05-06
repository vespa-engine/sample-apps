"""Export the passage encoder and the query encoder of a Hypencoder checkpoint
to ONNX, and save the tokenizer. Optionally INT8-quantize the query encoder
for ~3x faster CPU inference via ONNX Runtime's MLAS kernels.

Usage:
    python model2onnx.py --checkpoint jfkback/hypencoder.2_layer
    python model2onnx.py --checkpoint jfkback/hypencoder.2_layer --quantize-int8

Outputs (under ./app/models/, where services.xml references them):
    passage_encoder.onnx       Inlined-weight ONNX of the BERT-base passage encoder.
    query_encoder.onnx         Inlined-weight ONNX of the query encoder (BERT + hyperhead),
                               with multi-output (W0, b0, W1, b1, Wout) for the 2-layer q-net.
    tokenizer.json             HuggingFace fast-tokenizer JSON, used by both encoders.

The query encoder's wrapper performs a few important transforms so the rank
profile in schemas/doc.sd doesn't have to:
    * Casts float input_ids/attention_mask to int64 internally (Vespa cell
      types don't include int64, so we ship them as floats).
    * Pre-transposes weight matrices to (out, in) to match Vespa's
      alphabetical tensor-dimension ordering (the rank profile reads
      onnx(query_encoder).W0 as (d0=h, d1=x) - the in/out we want).
    * Uses static layer-norm shapes inside `_get_weights_and_biases` so the
      legacy TorchScript exporter inlines weights into a single .onnx file
      (Vespa filedistribution doesn't track external .data sidecars).
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from hypencoder_cb.modeling.hypencoder import (
    HypencoderDualEncoder,
    scaled_dot_product_attention,
)

OUT_DIR = Path("app/models")


# ============================================================================
# Passage encoder export (just BERT base; CLS pooling done by Vespa's HF embedder)
# ============================================================================
class PassageEncoderWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def export_passage_encoder(dual: HypencoderDualEncoder, out_path: Path) -> None:
    print(f"Exporting passage encoder to {out_path} ...")
    wrapped = PassageEncoderWrapper(dual.passage_encoder.transformer).eval()
    dummy_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.long)
    dummy_mask = torch.ones_like(dummy_ids)
    sidecar = out_path.with_suffix(out_path.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()
    torch.onnx.export(
        wrapped,
        (dummy_ids, dummy_mask),
        out_path.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    if sidecar.exists():
        sidecar.unlink()
    print(f"  size: {out_path.stat().st_size / 1e6:.1f} MB")


# ============================================================================
# Query encoder export (BERT + hyperhead, multi-output)
# ============================================================================
class QueryEncoderWrapper(torch.nn.Module):
    """Wraps the Hypencoder query encoder so its forward() returns the q-net
    weight tensors (W0, b0, W1, b1, Wout) directly instead of a callable
    NoTorchSequential. Internally reimplements `_get_weights_and_biases` with
    static layer-norm shapes (so the legacy ONNX exporter inlines weights)
    and pre-transposes Ws so the Vespa rank expression doesn't have to.
    """

    def __init__(self, hyp):
        super().__init__()
        self.hyp = hyp

    def _get_weights_and_biases_static(self, last_hidden_state, attention_mask):
        h = self.hyp
        batch_size = last_hidden_state.size(0)
        keys = [kp(last_hidden_state) for kp in h.key_projections]
        values = [vp(last_hidden_state) for vp in h.value_projections]
        weights_final, biases_final = [], []
        for i in range(len(h.weight_shapes)):
            w = scaled_dot_product_attention(
                query=h.weight_query_embeddings[i].repeat_interleave(batch_size, dim=0),
                key=keys[i], value=values[i],
                dim=h.weight_shapes[i][1], mask=attention_mask,
            )[0]
            w = F.layer_norm(F.relu(w), [h.weight_shapes[i][0]])
            w = h.weight_hyper_projection[i](w)
            w = (w + h.hyper_base_matrices[i].repeat(batch_size, 1, 1)).transpose(2, 1)
            weights_final.append(w)
        offset = len(h.weight_shapes)
        for i in range(len(h.bias_shapes)):
            b = scaled_dot_product_attention(
                query=h.bias_query_embeddings[i].repeat_interleave(batch_size, dim=0),
                key=keys[i + offset], value=values[i + offset],
                dim=h.bias_shapes[i][1], mask=attention_mask,
            )[0]
            b = F.layer_norm(F.relu(b), [h.bias_shapes[i][0]])
            b = h.bias_hyper_projection[i](b)
            b = (b + h.hyper_base_vectors[i].repeat(batch_size, 1, 1)).transpose(2, 1)
            biases_final.append(b)
        return weights_final, biases_final

    def forward(self, input_ids, attention_mask):
        # Vespa tensor cells are float/double/bfloat16/int8 (no int64), so
        # accept floats and cast.
        ids = input_ids.to(torch.long)
        mask = attention_mask.to(torch.long)
        out = self.hyp.transformer(input_ids=ids, attention_mask=mask)
        weights, biases = self._get_weights_and_biases_static(out.last_hidden_state, mask)
        # Squeeze batch + transpose Ws to (out, in) so the rank profile gets
        # them in the order Vespa stores tensor dims alphabetically (h, x).
        # Use explicit indexing rather than .squeeze() so ONNX Squeeze ops
        # only reduce dims that are statically size 1.
        # Last entry in `weights` is the final Wout (768->1, no bias).
        # All earlier entries pair with `biases` for residual blocks.
        n_blocks = len(biases)
        outs = []
        for i in range(n_blocks):
            outs.append(weights[i][0].transpose(0, 1).contiguous())  # W_i (out=768, in=768)
            outs.append(biases[i][0, :, 0].contiguous())              # b_i (out=768,)
        outs.append(weights[-1][0, :, 0].contiguous())                # Wout (in=768,)
        return tuple(outs)


def export_query_encoder(dual: HypencoderDualEncoder, out_path: Path) -> None:
    print(f"Exporting query encoder to {out_path} ...")
    wrapped = QueryEncoderWrapper(dual.query_encoder).eval()
    n_blocks = len(dual.query_encoder.bias_shapes)
    output_names = []
    for i in range(n_blocks):
        output_names += [f"W{i}", f"b{i}"]
    output_names.append("Wout")
    print(f"  q-net layers: {n_blocks} residual blocks + 1 final linear -> outputs {output_names}")
    dummy_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.float32)
    dummy_mask = torch.ones_like(dummy_ids)
    sidecar = out_path.with_suffix(out_path.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()
    torch.onnx.export(
        wrapped,
        (dummy_ids, dummy_mask),
        out_path.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    if sidecar.exists():
        sidecar.unlink()
    print(f"  size: {out_path.stat().st_size / 1e6:.1f} MB")


def quantize_query_encoder_int8(src: Path, dst: Path) -> None:
    """ORT dynamic int8 quantization of the BERT linears. Uses MLAS SMMLA
    on aarch64 with i8mm and VNNI on x86; ~3x faster BERT forward in our
    measurements with ~1% retrieval-quality drift. Static quantization with
    calibration data would do better."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print(f"INT8-quantizing {src} -> {dst} ...")
    quantize_dynamic(model_input=src.as_posix(), model_output=dst.as_posix(), weight_type=QuantType.QInt8)
    print(f"  size after quant: {dst.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="jfkback/hypencoder.2_layer",
                    help="HF model id of a Hypencoder checkpoint (default: jfkback/hypencoder.2_layer).")
    ap.add_argument("--out-dir", default="app/models",
                    help="Output directory for the ONNX files and tokenizer (default: ./app/models, "
                         "where services.xml expects them).")
    ap.add_argument("--quantize-int8", action="store_true",
                    help="After exporting, also run ORT dynamic INT8 quantization on the query encoder "
                         "and use that as the active query_encoder.onnx.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint} ...")
    dual = HypencoderDualEncoder.from_pretrained(args.checkpoint)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    dual.eval()

    export_passage_encoder(dual, out_dir / "passage_encoder.onnx")
    export_query_encoder(dual, out_dir / "query_encoder.onnx")

    tokenizer_path = out_dir / "tokenizer.json"
    tok.backend_tokenizer.save(tokenizer_path.as_posix())
    print(f"Saved tokenizer to {tokenizer_path}")

    if args.quantize_int8:
        fp32_path = out_dir / "query_encoder_fp32.onnx"
        shutil.move(out_dir / "query_encoder.onnx", fp32_path)
        quantize_query_encoder_int8(fp32_path, out_dir / "query_encoder.onnx")
        print(f"  (fp32 query encoder saved as {fp32_path} for reference)")


if __name__ == "__main__":
    main()
