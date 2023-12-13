
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Simple semantic search

A minimal semantic search application: 
- Query and document text is converted to embeddings by the application. 
- Search by embedding and/or text match.

<p data-test="run-macro init-deploy simple-semantic-search">
Requires at least Vespa 8.54.61.
</p>


## To try this

Follow
[vespa quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html)
through the <code>vespa deploy</code> step, cloning simple-semantic-search instead of album-recommendation.

Feed documents:
<pre data-test="exec">
vespa document ext/1.json
vespa document ext/2.json
vespa document ext/3.json
</pre>

Example queries:
<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(space contains many suns)"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(shipping stuff over the sea)"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e)" "input.query(e)=embed(exchanging information by sound)"
vespa query "yql=select * from doc where text contains 'boat'"
vespa query "yql=select * from doc where {targetHits: 100}nearestNeighbor(embedding, e) AND text contains 'boat'" "input.query(e)=embed(exchanging information by sound)"
</pre>

Remove the container after use:
<pre data-test="exec">
$ docker rm -f vespa
</pre>

## Ready for production

The [model included in this sample application](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
is suitable for production use and will produce good results in many domains without fine-tuning,
especially when combined with text match features such as bm25.

## Model exporting
Transformer based embedding models have named inputs and outputs that needs to be compatible with the input and output names used by the Bert embedder or the Huggingface embedder.

### Bert-embedder
See [export_model_from_hf.py](export_model_from_hf.py) for how to export a Huggingface sentence-transformer model to ONNX format compatible with default input and output names used by
the [bert-embedder](https://docs.vespa.ai/en/embedding.html#bert-embedder). 

The following exports [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) and saves the model parameters in a ONNX file and the `vocab.txt` file 
in the format expected by the Vespa bert-embedder.
<pre>
./export_model_from_hf.py --hf_model intfloat/e5-small-v2 --output_dir model
</pre>

### Huggingface-embedder
See [export_hf_model_from_hf.py](export_hf_model_from_hf.py) for how to export a Huggingface sentence-transformer model to ONNX format compatible with default input and output names used by
the [huggingface-embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder). 

The following exports [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) and saves the model parameters in a ONNX file and the `tokenizer.json` file. 
<pre>
./export_hf_model_from_hf.py --hf_model intfloat/e5-small-v2 --output_dir model
</pre>