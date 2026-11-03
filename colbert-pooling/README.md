# ColBERT with Hierarchical Token Pooling

> **Motivation.** ColBERT's multi-vector representation delivers excellent
> retrieval quality, but it comes with storage and query-time costs: every
> document token gets its own embedding, and every query needs per-token
> `nearestNeighbor` operators plus custom binarisation logic.
>
> This sample application prototypes the components that would make ColBERT a
> first-class citizen in Vespa: a pooling-aware embedder, a query searcher that
> handles all the multi-vector plumbing, and a two-phase ranking strategy that
> combines cheap HNSW retrieval with precise late-interaction reranking. The
> goal is to explore how these could be generalised and integrated into Vespa
> itself, so that using ColBERT becomes as simple as using any single-vector
> embedder.

## What this app does

All embedding, pooling, binarisation, query rewriting, and ranking runs
**inside Vespa** -- the user just sends text.

| Component | Role |
|---|---|
| `PoolingColBertEmbedder` | Runs the ONNX model at indexing time, optionally applies hierarchical token pooling (Ward's clustering), produces bfloat16 or int8 tensors |
| `ColBertSearcher` | At query time: embeds the query, binarises per-token vectors, builds `nearestNeighbor` OR query, sets all ranking tensors |
| Rank profile `colbert-pooled-ann` | HNSW ANN retrieval on binary pooled field, hamming MaxSim first phase, exact bfloat16 MaxSim second phase |

### Fields

| Field | Type | Storage | Description |
|---|---|---|---|
| `colbert` | `tensor<bfloat16>(dt{}, x[128])` | paged (disk) | Full non-pooled ColBERT embeddings for exact reranking |
| `colbert_pooled_binary` | `tensor<int8>(dt{}, x[16])` | in-memory + HNSW | Pooled + binarised embeddings for fast ANN retrieval |

### Memory savings

The combination of token pooling and binarisation dramatically reduces the
in-memory footprint of the HNSW-indexed field:

| Representation | Tokens (typical) | Bytes per token | Per-document (200 tokens) | 1B docs |
|---|---|---|---|---|
| Full bfloat16 | 200 | 256 (128 dims x 2B) | 50 KB | **~47 TB** |
| Pooled bfloat16 (factor=2) | 100 | 256 | 25 KB | ~24 TB |
| Pooled binary (factor=2) | 100 | 16 (128 bits packed) | 1.6 KB | ~1.5 TB |
| **Pooled binary (factor=4)** | **50** | **16** | **0.8 KB** | **~0.7 TB** |

With pool factor 4 and binarisation, the in-memory HNSW index is **~64x
smaller** than full bfloat16 -- making ColBERT-scale retrieval feasible at
billion-document scale. The full bfloat16 embeddings stay on disk (paged)
and are only read for the top-100 second-phase rerank.

### Ranking pipeline

| Phase | What | Field | Cost |
|---|---|---|---|
| **Retrieval** | `nearestNeighbor` ANN via HNSW (hamming) | `colbert_pooled_binary` (in-memory) | Cheap |
| **First-phase scoring** | Hamming-based MaxSim on binary embeddings | `colbert_pooled_binary` | Cheap |
| **Second-phase rerank** | Exact MaxSim on full bfloat16 (top 100) | `colbert` (paged from disk) | Expensive but precise |

## Why the custom searcher?

Vespa's `nearestNeighbor` operator works on single-vector query tensors, but
ColBERT queries are multi-vector (one embedding per token). To bridge this gap,
`ColBertSearcher` runs at query time and:

1. Calls the embedder to produce a float multi-vector query tensor (`qt`).
2. Binarises each token into a packed int8 vector.
3. Sets per-token `nearestNeighbor` query tensors (`rq0`..`rqN`) and ORs them
   together for HNSW retrieval.
4. Sets the binary multi-vector `qtb` for hamming MaxSim scoring.

This means the user just sends `query=planets in the solar system` and the
searcher handles all the plumbing. Without it, the client would need to
compute and pass 32+ binary tensors per request.

The per-token input declarations (`rq0`..`rq31`) in the rank profile are a
Vespa platform requirement: the content node needs them to resolve
`nearestNeighbor` terms, even though the searcher populates them
programmatically.

## Quick start

### 1. Export the ONNX model

```bash
uv pip install pylate-onnx-export onnxscript
uv run python export_model.py
```

This uses [pylate-onnx-export](https://github.com/lightonai/next-plaid/tree/main/next-plaid-onnx/python)
to export [ColBERT-Zero](https://huggingface.co/lightonai/ColBERT-Zero) to
ONNX with INT8 dynamic quantisation, and copies `model_int8.onnx` +
`tokenizer.json` into `src/main/application/models/`.

### 2. Build the application

```bash
mvn clean package -DskipTests
```

### 3. Deploy

```bash
vespa config set target local
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa

vespa deploy target/application
```

### 4. Feed 100 documents

```bash
vespa feed ext/feed.jsonl
```

Vespa computes both ColBERT representations at indexing time using the
`PoolingColBertEmbedder` component.

### 5. Query

Just send text -- the `ColBertSearcher` handles embedding, binarisation, and
`nearestNeighbor` query construction:

```bash
vespa query 'query=planets in the solar system'
vespa query 'query=machine learning and artificial intelligence'
vespa query 'query=renewable energy sources'
```

### 6. Run tests

```bash
mvn test
```

### 7. Verify correctness

The `verify_correctness.py` script validates Vespa's embeddings against
[pylate](https://github.com/lightonai/pylate), the model authors' reference
implementation.

```
pylate (FP32 PyTorch)  ←  quantisation floor (cos ~0.98)  →  ONNX INT8
                                                               ↕ identical (cos >0.999)
                                                             Vespa (Java + ONNX INT8)
```

Vespa and the Python ONNX INT8 reference produce near-identical embeddings
(cos_sim > 0.999 per token, 100% bit-exact binarisation). The ~2% gap to
pylate FP32 is the INT8 quantisation floor, not an implementation difference.

```bash
uv pip install pylate onnxruntime tokenizers

# Full comparison (Vespa must be running):
uv run python verify_correctness.py

# Without Vespa (pylate vs ONNX INT8 only):
uv run python verify_correctness.py --no-vespa
```

## Hierarchical token pooling

The `HierarchicalTokenPooling` class (invoked by `PoolingColBertEmbedder`
after the ONNX model produces per-token embeddings) implements:

1. **Pairwise cosine distances** between all token embeddings (excluding CLS).
2. **Ward's agglomerative clustering** using the Nearest-Neighbor Chain (NNC)
   algorithm -- O(n^2) time.
3. **Cut the dendrogram** to `ceil(n / poolFactor)` clusters.
4. **Replace each cluster with its L2-normalised centroid**.
5. **Prepend the original CLS token** (always preserved).

The algorithm is a Java port of the
[hierarchy.rs](https://github.com/lightonai/next-plaid/blob/main/next-plaid-onnx/src/hierarchy.rs)
module from [next-plaid-onnx](https://github.com/lightonai/next-plaid).

### Performance (Java, 128-dim embeddings)

| Tokens | `poolTokens` | `pdistCosine` |
|--------|-------------|---------------|
| 128    | ~1 ms       | < 1 ms        |
| 512    | ~14 ms      | ~9 ms         |

## Architecture

### Embedder

`PoolingColBertEmbedder` is a custom Vespa `Embedder` that runs the ColBERT
ONNX model and optionally applies hierarchical token pooling, controlled by
`poolFactor`:

- **`poolFactor=0`** -- standard ColBERT (no pooling)
- **`poolFactor=2`** -- merge similar tokens, keep ~half the vectors
- **`poolFactor=3`** -- keep roughly a third, etc.

Two instances are configured in `services.xml` with different `poolFactor`
values, sharing the same ONNX model and tokenizer.

### Sequence construction

Input sequences match [pylate](https://github.com/lightonai/pylate)'s
ColBERT encoding:

```
[CLS] [D] search_document: <document text> [SEP]    (documents)
[CLS] [Q] search_query: <query text> [MASK]...       (queries, padded)
```

The `search_document:` / `search_query:` prefixes are configurable via
`prependDocument` / `prependQuery` in the
[config definition](src/main/resources/configdefinitions/pooling-colbert-embedder.def).
ColBERT-Zero requires these prompts for full retrieval quality
([model card](https://huggingface.co/lightonai/ColBERT-Zero)).

## Project structure

```
colbert-pooling/
├── pom.xml                                             # Maven build (container-plugin)
├── src/main/
│   ├── application/                                    # Vespa application package
│   │   ├── schemas/doc.sd                              # Schema, HNSW index, rank profile
│   │   ├── services.xml                                # Embedder + Searcher config
│   │   └── search/query-profiles/                      # Query tensor type definitions
│   ├── java/ai/vespa/colbert/
│   │   ├── HierarchicalTokenPooling.java               # Ward NNC + fcluster + poolTokens
│   │   ├── PoolingColBertEmbedder.java                 # Custom Vespa Embedder
│   │   └── ColBertSearcher.java                        # Query rewriting + embedding
│   └── resources/configdefinitions/
│       └── pooling-colbert-embedder.def                # Embedder config definition
├── src/test/
│   ├── java/ai/vespa/colbert/
│   │   └── HierarchicalTokenPoolingTest.java           # JUnit tests + benchmarks
│   └── application/tests/system-test/
│       └── feed-and-search-test.json                   # Vespa system test
├── verify_correctness.py                               # E2E: pylate vs ONNX INT8 vs Vespa
├── ext/feed.jsonl                                      # 100 sample documents
├── export_model.py                                     # ONNX export via pylate-onnx-export
└── feed.py                                             # Feed JSONL generation helper
```
