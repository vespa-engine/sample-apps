
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Binary Closeness

This sample application demonstrates semantic search using binary (quantized) embeddings with hamming distance.
It showcases two different closeness calculations:

1. **Standard closeness**: `1 / (1 + hamming_distance)` - Vespa's built-in closeness function
2. **Normalized binary closeness (similarity)**: `1 - (hamming_distance / 768)` - Normalized to [0, 1] range

The application uses the [Alibaba GTE-ModernBERT](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) model
with `pack_bits` to create 96-dimensional int8 binary embeddings (768 bits total).

## Deploy

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started)
through the `vespa deploy` step, cloning `binary-closeness` instead of `album-recommendation`.

Or deploy locally with Docker:

```bash
# Start Vespa container
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa

# Wait for config server
vespa status deploy --wait 300

# Deploy the application
vespa deploy app
```

## Feed documents

Feed the sample documents (embedding inference happens in Vespa):

```bash
vespa feed dataset/documents.jsonl
```

## Query

Run a semantic search query with both closeness metrics returned:

```bash
vespa query \
  'yql=select * from music where {targetHits: 10}nearestNeighbor(embedding, q)' \
  'input.query(q)=embed(alibaba_gte_modernbert, @query)' \
  'query=british rock band' \
  'ranking.profile=semantic'
```

The response will include `summaryfeatures` with both closeness calculations:
- `closeness_score`: Standard closeness (1 / (1 + distance))
- `similarity`: Normalized binary closeness (1 - distance/768)

Example response snippet:
```json
{
  "fields": {
    "text": "Coldplay is a British rock band...",
    "summaryfeatures": {
      "closeness_score": 0.0625,
      "similarity": 0.85
    }
  }
}
```

## Understanding the closeness metrics

For binary embeddings with hamming distance:

- **Hamming distance**: Number of differing bits between two binary vectors (0 to 768 for 96-dim int8)
- **closeness_score**: `1 / (1 + hamming_distance)` - Non-linear, ranges from 1 (identical) to ~0.0013 (max distance)
- **similarity**: `1 - (hamming_distance / 768)` - Linear, ranges from 1 (identical) to 0 (max distance)

The `similarity` metric is often more intuitive as it maps directly to the percentage of matching bits.

## Clean up

Remove the container after use:

```bash
docker rm -f vespa
```
