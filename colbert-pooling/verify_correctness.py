#!/usr/bin/env python3
"""End-to-end correctness verification: Vespa vs pylate (model authors' reference).

Both produce [CLS] [D] search_document: <text> [SEP] sequences.  pylate runs
FP32 PyTorch; Vespa runs the INT8 ONNX model.  The expected cos_sim floor is
~0.94 per token (INT8 quantisation).

Prerequisites:
    uv pip install pylate onnxruntime tokenizers requests numpy

Usage:
    uv run python verify_correctness.py                  # full comparison
    uv run python verify_correctness.py --no-vespa       # pylate vs ONNX INT8 only
"""

from __future__ import annotations

import argparse
import string
import sys
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path("src/main/application/models")
VESPA_ENDPOINT = "http://localhost:8080"

CLS, SEP, D_TOKEN = 50281, 50282, 50369
PREPEND_DOC = "search_document: "

TEST_DOC = (
    "The solar system consists of the Sun and the celestial bodies that "
    "orbit it, including eight planets, their moons, dwarf planets, "
    "asteroids, and comets. The four inner planets are rocky worlds, "
    "while the outer planets are gas and ice giants."
)

POOL_FACTORS = [2, 3, 4]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if min(na, nb) > 1e-10 else 0.0


def build_skip(tok_nospc):
    skip = set()
    for ch in string.punctuation:
        skip.update(tok_nospc.encode(ch).ids)
    return skip


def onnx_embed(text, tok_nospc, session, skip):
    """Same sequence as Vespa PoolingColBertEmbedder: [CLS, D, filtered_tokens, SEP]."""
    enc = tok_nospc.encode(PREPEND_DOC + text)
    tids = [t for t in enc.ids if t not in skip][:512 - 3]
    ids = [CLS, D_TOKEN] + tids + [SEP]
    mask = [1] * len(ids)
    (out,) = session.run(None, {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.array([mask], dtype=np.int64),
    })
    return out[0, :len(ids)].astype(np.float32)


def compare(name, ref, test, min_cos):
    if ref.shape != test.shape:
        print(f"  FAIL {name}: shape {ref.shape} vs {test.shape}")
        return False
    sims = np.array([cos(ref[i], test[i]) for i in range(len(ref))])
    ok = sims.min() >= min_cos
    print(f"  {'PASS' if ok else 'FAIL'} {name}: {len(ref)} tokens, "
          f"cos mean={sims.mean():.6f} min={sims.min():.6f}")
    return ok


# ── Pooling (matches Java HierarchicalTokenPooling) ─────────────────────────

def _pdist(emb):
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-10)
    s = (emb / norms) @ (emb / norms).T
    r, c = np.triu_indices(len(emb), k=1)
    return np.clip(1.0 - s[r, c], 0.0, 2.0).astype(np.float64)

def _ward(d, n):
    tot = 2*n-1; D = np.full((tot,tot), np.inf); idx=0
    for i in range(n):
        for j in range(i+1,n): d2=float(d[idx])**2; D[i,j]=d2; D[j,i]=d2; idx+=1
    sz=np.ones(tot,dtype=np.int64); act=np.zeros(tot,dtype=bool); act[:n]=True
    Z=np.empty((n-1,4)); ch=[]
    for s in range(n-1):
        if not ch: ch.append(int(np.argmax(act)))
        while True:
            r=D[ch[-1]].copy(); r[~act]=np.inf; r[ch[-1]]=np.inf; nn=int(np.argmin(r))
            if len(ch)>=2 and nn==ch[-2]: b=ch.pop(); a=ch.pop(); break
            ch.append(nn)
        if a>b: a,b=b,a
        md=D[a,b]; nid=n+s; Z[s]=[a,b,np.sqrt(max(md,0)),sz[a]+sz[b]]; sz[nid]=sz[a]+sz[b]
        act[a]=False; act[b]=False; ai=np.where(act)[0]
        if len(ai):
            na,nb=float(sz[a]),float(sz[b]); nk=sz[ai].astype(np.float64)
            dn=((na+nk)*D[a,ai]+(nb+nk)*D[b,ai]-nk*md)/(na+nb+nk)
            D[nid,ai]=dn; D[ai,nid]=dn
        act[nid]=True
    return Z

def _fclust(Z,n,k):
    k=max(1,min(k,n)); nm=n-k
    if nm==0: return np.arange(1,n+1,dtype=np.int64)
    p=list(range(2*n-1))
    def f(x):
        while p[x]!=x: p[x]=p[p[x]]; x=p[x]
        return x
    si=np.argsort(Z[:,2])
    for m in range(nm): i=int(si[m]); a,b=int(Z[i,0]),int(Z[i,1]); p[f(a)]=n+i; p[f(b)]=n+i
    lb=np.empty(n,dtype=np.int64); cm={}; nl=1
    for i in range(n):
        r=f(i)
        if r not in cm: cm[r]=nl; nl+=1
        lb[i]=cm[r]
    return lb

def pool(emb, pf):
    e=emb.astype(np.float64); cls=e[0:1]; t=e[1:]; n=len(t)
    if n<=1 or pf<=1: return e.astype(np.float32)
    nc=max(1,int(np.ceil(n/pf)))
    if nc>=n: return e.astype(np.float32)
    Z=_ward(_pdist(t),n); lb=_fclust(Z,n,nc)
    po=np.empty((len(np.unique(lb)),t.shape[1]),dtype=np.float64)
    for i,l in enumerate(np.unique(lb)):
        c=t[lb==l].mean(axis=0); po[i]=c/max(np.linalg.norm(c),1e-10)
    return np.vstack([cls,po]).astype(np.float32)

def binarize(emb):
    e=emb.astype(np.float64); n,d=e.shape
    b=(e>0).astype(np.uint8).reshape(n,d//8,8)
    return np.sum(b*np.array([128,64,32,16,8,4,2,1],dtype=np.uint8),axis=2).astype(np.uint8).view(np.int8)


# ── Vespa ────────────────────────────────────────────────────────────────────

def vespa_feed(ep, did, text):
    import requests
    requests.post(f"{ep}/document/v1/doc/doc/docid/{did}",
                  json={"fields":{"doc_id":did,"title":"test","text":text}}, timeout=60).raise_for_status()

def vespa_tensors(ep, did):
    import requests
    r = requests.post(f"{ep}/search/", json={
        "yql": f'select * from doc where doc_id contains "{did}"',
        "hits":1, "summary":"tensors"}, timeout=30)
    r.raise_for_status()
    return r.json()["root"]["children"][0]["fields"]

def parse_tensor(tj):
    if "blocks" in tj:
        ks=sorted(tj["blocks"],key=int); return np.array([tj["blocks"][k] for k in ks],dtype=np.float64)
    cells=tj["cells"]
    mt=max(int(c["address"]["dt"]) for c in cells)+1; mx=max(int(c["address"]["x"]) for c in cells)+1
    a=np.zeros((mt,mx),dtype=np.float64)
    for c in cells: a[int(c["address"]["dt"]),int(c["address"]["x"])]=c["value"]
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=VESPA_ENDPOINT)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--no-vespa", action="store_true")
    args = p.parse_args()

    model_path = args.model_dir / "model_int8.onnx"
    tok_path = args.model_dir / "tokenizer.json"
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    ok = True

    # ── pylate FP32 reference ────────────────────────────────────────────
    print("Loading pylate (FP32 PyTorch) ...")
    from pylate import models as pm
    pylate_model = pm.ColBERT("lightonai/ColBERT-Zero", device="cpu")
    pylate_emb = np.array(pylate_model.encode([TEST_DOC], is_query=False, prompt_name="document")[0], dtype=np.float32)
    print(f"  pylate: {pylate_emb.shape}")

    # ── ONNX INT8 reference (same model file as Vespa) ───────────────────
    print("Loading ONNX INT8 model ...")
    import onnxruntime as ort
    from tokenizers import Tokenizer
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    tok_nospc = Tokenizer.from_file(str(tok_path))
    tok_nospc.no_padding(); tok_nospc.no_truncation(); tok_nospc.post_processor = None
    skip = build_skip(tok_nospc)

    onnx_emb = onnx_embed(TEST_DOC, tok_nospc, session, skip)
    print(f"  ONNX INT8: {onnx_emb.shape}")

    # ── Step 1: pylate vs ONNX INT8 (quantisation floor) ─────────────────
    print(f"\n{'='*72}")
    print("Step 1: pylate FP32 vs ONNX INT8 (quantisation floor)")
    print(f"{'='*72}")
    ok &= compare("pylate vs ONNX INT8", pylate_emb, onnx_emb, min_cos=0.90)

    # ── Step 2: Pooling (Python reference, various factors) ──────────────
    print(f"\n{'='*72}")
    print("Step 2: Pooling correctness (Python, on ONNX INT8 embeddings)")
    print(f"{'='*72}")
    pooled_ref = {}
    for pf in POOL_FACTORS:
        po = pool(onnx_emb, pf)
        n_in, n_out = onnx_emb.shape[0], po.shape[0]
        exp = 1 + int(np.ceil((n_in - 1) / pf))
        ok_s = n_out == exp
        ok_c = np.allclose(po[0], onnx_emb[0], atol=1e-6)
        ok_n = np.allclose(np.linalg.norm(po, axis=1), 1.0, atol=1e-6)
        stat = "PASS" if (ok_s and ok_c and ok_n) else "FAIL"
        if stat == "FAIL": ok = False
        print(f"  {stat} factor={pf}: {n_in} -> {n_out} (exp {exp}), CLS={'ok' if ok_c else 'FAIL'}, norms ok={ok_n}")
        pooled_ref[pf] = po
    bin_ref = binarize(pooled_ref[2])

    # ── Step 3: Vespa vs ONNX INT8 + Python pooling ─────────────────────
    if args.no_vespa:
        print(f"\n{'='*72}")
        print("Step 3: Vespa comparison SKIPPED (--no-vespa)")
        print(f"{'='*72}")
    else:
        print(f"\n{'='*72}")
        print(f"Step 3: Vespa vs ONNX INT8 reference ({args.endpoint})")
        print(f"{'='*72}")
        try:
            import requests
            did = "verify_test_0"
            vespa_feed(args.endpoint, did, TEST_DOC)
            time.sleep(1)
            f = vespa_tensors(args.endpoint, did)

            vc = parse_tensor(f["colbert"])
            print(f"  Vespa colbert: {vc.shape}")
            ok &= compare("non-pooled (ONNX INT8 vs Vespa)", onnx_emb.astype(np.float64), vc, min_cos=0.999)

            vp = parse_tensor(f["colbert_pooled"])
            rp = pooled_ref[2]
            print(f"  Vespa colbert_pooled: {vp.shape}, ref: {rp.shape}")
            if vp.shape == rp.shape:
                ok &= compare("pooled factor=2 (Python vs Vespa)", rp.astype(np.float64), vp, min_cos=0.999)
            else:
                print(f"  FAIL shape mismatch"); ok = False

            vb = parse_tensor(f["colbert_pooled_binary"]).astype(np.int8)
            m = int(np.sum(vb == bin_ref)); t = vb.size; p = 100*m/t
            stat = "PASS" if p == 100 else "FAIL"
            if stat == "FAIL": ok = False
            print(f"  {stat} binary: {m}/{t} bytes match ({p:.1f}%)")

            # Also compare Vespa directly against pylate
            print()
            ok &= compare("non-pooled (pylate FP32 vs Vespa)", pylate_emb.astype(np.float64), vc, min_cos=0.90)

            requests.delete(f"{args.endpoint}/document/v1/doc/doc/docid/{did}", timeout=10)
        except Exception as e:
            if "Connection" in str(type(e).__name__) or "Connection" in str(e):
                print(f"  SKIP: cannot connect to {args.endpoint}")
            else:
                print(f"  ERROR: {e}"); ok = False

    print(f"\n{'='*72}")
    print(f"RESULT: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print(f"{'='*72}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
