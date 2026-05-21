"""Helper: emit the JSON body to send to /search/ for each rank profile.

Four modes:

  --cosine      For the `cosine_baseline` rank profile. Sends only the query
                text - Vespa runs the passage embedder server-side via embed()
                to produce the query vector.

  --rerank      For the `hypencoder_rerank` rank profile. Sends the query text
                (server-side embed() for the cosine first phase) AND the
                tokenized query (for the ONNX q-net second phase).

  --lexical     For the `hypencoder_lexical_rerank` rank profile. BM25
                first-phase against the text field, hypencoder q-net on the
                top candidates. Sends `userQuery()` + tokenized query; no
                vector.

  default       For the `hypencoder_onnx` rank profile (rank-all). Sends only
                token IDs + attention mask. ~830 byte payload.

All modes are designed to be piped into `vespa query --file`:

    python encode_query.py "tallest mountain in the world" > q.json
    vespa query --file q.json
"""
import argparse
import json
import sys

import numpy as np
from transformers import AutoTokenizer

MAX_SEQ = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="The query text.")
    ap.add_argument("--checkpoint", default="jfkback/hypencoder.2_layer")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--rerank", action="store_true",
                      help="Emit a body for hypencoder_rerank.")
    mode.add_argument("--cosine", action="store_true",
                      help="Emit a body for cosine_baseline.")
    mode.add_argument("--lexical", action="store_true",
                      help="Emit a body for hypencoder_lexical_rerank.")
    ap.add_argument("--hits", type=int, default=10)
    args = ap.parse_args()

    body = {
        "yql": "select id, text from doc where true",
        "hits": args.hits,
        "timeout": "30s",
    }

    if args.cosine:
        body["ranking.profile"] = "cosine_baseline"
        body["input.query(q_vec)"] = "embed(passage_embedder, @q)"
        body["q"] = args.query
    else:
        tok = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
        enc = tok([args.query], padding="max_length", truncation=True,
                  max_length=MAX_SEQ, return_tensors="np")
        body["input.query(input_ids)"] = [enc["input_ids"][0].astype(np.float32).tolist()]
        body["input.query(attention_mask)"] = [enc["attention_mask"][0].astype(np.float32).tolist()]
        if args.rerank:
            body["ranking.profile"] = "hypencoder_rerank"
            body["input.query(q_vec)"] = "embed(passage_embedder, @q)"
            body["q"] = args.query
        elif args.lexical:
            body["yql"] = ('select id, text from doc where '
                           '{grammar: "weakAnd", defaultIndex: "text"}userInput(@q)')
            body["q"] = args.query
            body["ranking.profile"] = "hypencoder_lexical_rerank"
        else:
            body["ranking.profile"] = "hypencoder_onnx"

    json.dump(body, sys.stdout)


if __name__ == "__main__":
    main()
