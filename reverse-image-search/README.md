<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample application - Reverse Image Search

## Short description
Reverse Image Search takes an input **image** and returns visually similar images from an indexed
corpus. Embeddings come from [DINOv2](https://github.com/facebookresearch/dinov2)
(`facebook/dinov2-base`, 768-d CLS, L2-normalized). The schema stores both the float embedding
(used for second-phase rerank) and a server-side derived binary embedding (used for cheap HNSW
first-phase retrieval over Hamming distance).

The original images are inlined inside Vespa as base64 `raw` bytes (`full_image`), so the index is
the single source of truth for vectors, metadata, and image bytes.


## Features
- DINOv2 (`facebook/dinov2-base`) image embeddings — 768-d, L2-normalized.
- **Binary HNSW** first-phase retrieval over a server-derived `tensor<int8>(x[96])` field
  (768 signed bits packed into 96 int8s via `| binarize | pack_bits`), Hamming distance.
- **Float rerank** in second phase using a `tensor<bfloat16>(x[768])` `paged` attribute —
  ~16× less memory than an HNSW-indexed in-memory float attribute.
- Five rank profiles (`closeness`, `closeness_hybrid_strict`, `closeness_binary`,
  `closeness_weighted`, plus `unranked`/`random`) so you can compare hybrid binary+float retrieval
  against binary-only or weighted blends.
- JPEG bytes inlined as `full_image` — Vespa serves both the search index and the image payload.


## Quick start
The dataset is [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) (1.28M
train + 50K validation images), streamed from HuggingFace, embedded with DINOv2 on the client,
then fed to Vespa with the JPEG bytes inlined.

Requirements:
* [Docker](https://www.docker.com/) Desktop installed and running. 8 GB available memory for Docker
  is recommended for a small subset; the full corpus needs Vespa Cloud (see
  [Vespa Cloud notes](#vespa-cloud-notes)).
  Refer to [Docker memory](https://docs.vespa.ai/en/operations-selfhosted/docker-containers.html#memory)
  for details and troubleshooting.
* Operating system: Linux, macOS or Windows 10 Pro (Docker requirement).
* Architecture: x86_64 or arm64.
* [Homebrew](https://brew.sh/) to install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html), or
  download a Vespa CLI release from [GitHub releases](https://github.com/vespa-engine/vespa/releases).
* Python 3.11+ with `pip` (for the feeder and the search CLI).
* A HuggingFace token with access to `ILSVRC/imagenet-1k` (gated dataset). Set it as `HF_TOKEN`
  before running the feeder.

Validate Docker resource settings, should be minimum 8 GB:
<pre>
$ docker info | grep "Total Memory"
or
$ podman info | grep "memTotal"
</pre>

Install [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):
<pre>
$ brew install vespa-cli
</pre>

For local deployment using docker image:
<pre data-test="exec">
$ vespa config set target local
</pre>

Pull and start the Vespa Docker container:
<pre data-test="exec">
$ docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 vespaengine/vespa
</pre>

Clone this sample app:
<pre data-test="exec">
$ vespa clone reverse-image-search myapp && cd myapp
</pre>

Wait for the configserver to start:
<pre data-test="exec" data-test-assert-contains="is ready">
$ vespa status deploy --wait 300 --color never
</pre>

Deploy the application and wait for services to start:
<pre data-test="exec">
$ vespa deploy app --wait 300 --color never
</pre>

> [!NOTE]
> The bundled `app/services.xml` requests 2 container nodes and 2 content nodes with explicit
> `<resources>`. That sizing is intended for Vespa Cloud (the corpus inlines ~75 GB of JPEG bytes
> for the full 1.28M train split). For a local single-container deploy, edit `app/services.xml`
> and change both `<nodes count="2">` blocks to `count="1"` and remove the `<resources>` element
> before running `vespa deploy app`.

## Indexing
Install the Python dependencies for the feeder and search CLI:
<pre>
$ pip install -r script/requirements.txt
</pre>

Authenticate with HuggingFace (the dataset is gated):
<pre>
$ export HF_TOKEN=hf_...
</pre>

Point the feeder at your Vespa endpoint and feed a small smoke-test subset (500 validation
images — minutes on CPU):
<pre>
$ export RIS_VESPA_URL=http://localhost:8080
$ python script/feed.py --split validation --limit 500
</pre>

Other useful invocations:
<pre>
$ python script/feed.py --split validation              # full validation split (~50K)
$ python script/feed.py --split train                   # full train split (~1.28M, GPU strongly recommended)
$ python script/feed.py --split validation --refeed     # re-feed images already on disk (use after schema changes)
</pre>

The feeder streams from HuggingFace, resizes each image to max-side 1024 px, embeds with DINOv2,
base64-encodes the JPEG bytes, and feeds each doc to Vespa with both the bfloat16 embedding and
the inlined image. It is resumable — IDs whose JPEG is already on disk are skipped unless
`--refeed` is set.

## Querying
Run a reverse-image-search query from the command line:
<pre>
$ python script/search.py path/to/query.jpg --hits 10 --ranking hybrid
</pre>

Available rank profiles (mirrored from `script/search.py`):

| `--ranking` | Schema rank profile | What it does |
|---|---|---|
| `hybrid` (default) | `closeness` | Binary HNSW first-phase, float cosine rerank on top 100 |
| `hybrid_strict` | `closeness_hybrid_strict` | Same first-phase, deeper float rerank (top 1000) |
| `binary` | `closeness_binary` | Binary HNSW only, no rerank — cheapest, smallest memory |
| `weighted` | `closeness_weighted` | Binary HNSW first-phase, second-phase = α·float + (1−α)·binary |

Both `embedding` and `embedding_binary` are surfaced in the result's `matchfeatures` so you can
compare the binary closeness, the manually-computed float closeness (`closeness_emb`), and the
implied bit-agreement ratio for every hit, regardless of which profile was used.

## How it works
- **Indexing.** Each image is embedded client-side with DINOv2 → L2-normalized 768-d float vector,
  stored as `tensor<bfloat16>(x[768])` with `paged` attribute (on disk). Vespa derives the binary
  embedding server-side via `indexing: input embedding | binarize | pack_bits | attribute | index`,
  giving a `tensor<int8>(x[96])` HNSW-indexed in-memory attribute with Hamming distance.
- **Retrieval.** Queries run `nearestNeighbor(embedding_binary, q_bin)` for cheap first-phase
  retrieval. The float `embedding` is read from the paged attribute only for the rerank-count
  candidates that survive first-phase, so per-query disk I/O is bounded.
- **Cosine via dot product.** When retrieval is binary-driven, Vespa does **not** populate the
  built-in `closeness(field, embedding)` cache for candidates, so the schema computes float
  cosine manually as `sum(query(q) * attribute(embedding))` (both sides L2-normalized) and
  surfaces it as `closeness_emb` for second-phase ranking and match-features.
- **Image serving.** `full_image` (base64-encoded JPEG bytes) is fetched on demand via the
  `with-image` document summary so the index is the single source of truth — there is no
  separate image store.

## Vespa Cloud notes
This app is sized for Vespa Cloud out of the box (`app/deployment.xml` pins
`aws-us-east-1c`, `app/services.xml` requests 2× content nodes at 16 GB / 200 GB). To deploy to
Vespa Cloud:

1. `vespa auth login`
2. `vespa config set target cloud`
3. `vespa config set application <tenant>.reverse-image-search.default`
4. `vespa auth cert -N`
5. Copy the generated cert+key from `~/.vespa/<tenant>.reverse-image-search.default/` and point
   `RIS_VESPA_CERT_PATH` and `RIS_VESPA_KEY_PATH` at them.
6. `vespa deploy app --wait 600` — note the printed mTLS endpoint URL.
7. `export RIS_VESPA_URL=https://<endpoint>.vespa-app.cloud` before running the feeder or
   `script/search.py`.

See the [Vespa Cloud getting-started guide](https://docs.vespa.ai/en/cloud/getting-started.html)
for full details.

## Cleanup
Shutdown and remove the local Vespa container:
<pre data-test="after">
$ docker rm -f vespa
</pre>
