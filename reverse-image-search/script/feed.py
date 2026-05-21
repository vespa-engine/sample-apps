"""Feed ImageNet-1k into Vespa.

Streams ILSVRC/imagenet-1k from HuggingFace, saves JPEGs to IMAGES_DIR,
generates DINOv2 embeddings, and feeds to Vespa with bfloat16 tensors.

Resumable: skips IDs whose image file is already present on disk.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
from pathlib import Path
from typing import Iterable, Iterator

import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from vespa.application import Vespa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("feeder")

VESPA_URL = os.environ["RIS_VESPA_URL"].rstrip("/")
VESPA_CERT_PATH = os.environ.get("RIS_VESPA_CERT_PATH") or None
VESPA_KEY_PATH = os.environ.get("RIS_VESPA_KEY_PATH") or None
IMAGES_DIR = Path(os.environ.get("RIS_IMAGES_DIR", "/data/images"))
MODEL_NAME = os.environ.get("RIS_MODEL_NAME", "facebook/dinov2-base")
HF_DATASET = os.environ.get("RIS_HF_DATASET", "ILSVRC/imagenet-1k")
BATCH_SIZE = int(os.environ.get("RIS_BATCH_SIZE", "32"))
FEED_CONCURRENCY = int(os.environ.get("RIS_FEED_CONCURRENCY", "8"))


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _save_image(img: Image.Image, path: Path, max_side: int = 1024) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(path, format="JPEG", quality=88)
    return img.size


def _make_id(split: str, index: int) -> str:
    return f"{split}-{index:08d}"


def _relative_path(split: str, index: int) -> str:
    shard = index // 1000
    return f"{split}/{shard:04d}/{_make_id(split, index)}.jpg"


def _iter_split(split: str, limit: int | None) -> Iterator[tuple[int, dict]]:
    log.info("streaming split=%s limit=%s", split, limit)
    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield i, ex


def _batched(iterable: Iterable, size: int) -> Iterator[list]:
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_docs_for_feed(docs: list[dict]) -> Iterator[dict]:
    for d in docs:
        yield {
            "id": d["id"],
            "fields": {
                "id": d["id"],
                "filename": d["filename"],
                "path": d["path"],
                "split": d["split"],
                "label": d["label"],
                "width": d["width"],
                "height": d["height"],
                "embedding": {"values": d["embedding"]},
                "full_image": d["full_image"],
            },
        }


def run(split: str, limit: int | None, refeed: bool = False) -> None:
    device = _device()
    log.info("loading %s on %s", MODEL_NAME, device)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    vespa_kwargs: dict = {"url": VESPA_URL}
    if VESPA_CERT_PATH and VESPA_KEY_PATH:
        vespa_kwargs["cert"] = VESPA_CERT_PATH
        vespa_kwargs["key"] = VESPA_KEY_PATH
    app = Vespa(**vespa_kwargs)

    total_ok = 0
    total_fail = 0
    total_skip = 0

    def callback(response, doc_id):
        nonlocal total_ok, total_fail
        if response.is_successful():
            total_ok += 1
        else:
            total_fail += 1
            log.warning("feed fail id=%s status=%s body=%s",
                        doc_id, response.get_status_code(), response.get_json())

    with torch.inference_mode():
        for batch in _batched(_iter_split(split, limit), BATCH_SIZE):
            pil_images: list[Image.Image] = []
            pending: list[dict] = []

            for i, ex in batch:
                doc_id = _make_id(split, i)
                rel = _relative_path(split, i)
                abs_path = IMAGES_DIR / rel
                img = ex["image"]
                if not isinstance(img, Image.Image):
                    continue
                # Save if missing; otherwise skip unless --refeed.
                if abs_path.exists() and not refeed:
                    total_skip += 1
                    continue
                try:
                    if not abs_path.exists():
                        w, h = _save_image(img, abs_path)
                    else:
                        with Image.open(abs_path) as existing:
                            w, h = existing.size
                except Exception as e:
                    log.warning("save failed id=%s: %s", doc_id, e)
                    continue
                jpeg_bytes = abs_path.read_bytes()
                pil_images.append(Image.open(abs_path).convert("RGB"))
                pending.append({
                    "id": doc_id,
                    "filename": f"{doc_id}.jpg",
                    "path": rel,
                    "split": split,
                    "label": int(ex.get("label", -1) or -1),
                    "width": w,
                    "height": h,
                    "full_image": base64.b64encode(jpeg_bytes).decode("ascii"),
                })

            if not pending:
                continue

            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            output = model(**inputs)
            cls = output.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
            embs = cls.to("cpu", dtype=torch.float32).numpy()

            for rec, emb in zip(pending, embs):
                rec["embedding"] = emb.tolist()

            # feed_iterable (sync, threaded) is used instead of feed_async_iterable
            # because pyvespa 1.1.2's async feed path does not propagate mTLS certs
            # to its internal httpr client — every request gets a 401 from Vespa Cloud.
            app.feed_iterable(
                iter=_iter_docs_for_feed(pending),
                schema="image",
                callback=callback,
                max_queue_size=FEED_CONCURRENCY * 8,
                max_workers=FEED_CONCURRENCY,
                max_connections=FEED_CONCURRENCY,
            )
            log.info("progress split=%s ok=%d fail=%d skip=%d",
                     split, total_ok, total_fail, total_skip)

    log.info("done split=%s ok=%d fail=%d skip=%d",
             split, total_ok, total_fail, total_skip)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="cap number of images (for testing)")
    parser.add_argument(
        "--refeed",
        action="store_true",
        help="Re-feed images that are already on disk (use after a schema change).",
    )
    args = parser.parse_args()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    run(args.split, args.limit, refeed=args.refeed)


if __name__ == "__main__":
    main()
