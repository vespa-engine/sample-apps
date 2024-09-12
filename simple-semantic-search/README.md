
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - Simple semantic search

A minimal semantic search application: 
- Query and document text is converted to embeddings by the application using Vespa's [embedder functionality](https://docs.vespa.ai/en/embedding.html#huggingface-embedder).
- Search by embedding or text match and use [reciprocal rank fusion](https://docs.vespa.ai/en/phased-ranking.html#cross-hit-normalization-including-reciprocal-rank-fusion) to fuse 
different rankings.

<p data-test="run-macro init-deploy simple-semantic-search">
minimum-required-vespa-version="8.311.28"
</p>

## To try this application

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started)
through the <code>vespa deploy</code> step, cloning `simple-semantic-search` instead of `album-recommendation`.

Feed documents (this includes embed inference in Vespa):

<pre data-test="exec">
vespa feed ext/*.json
</pre>

Example queries using [E5-Small-V2](https://huggingface.co/intfloat/e5-small-v2) 
embedding model that maps text to a 384-dimensional vector representation.

<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, e))' \
 'input.query(e)=embed(e5, @query)' \
 'query=space contains many suns'
</pre>

<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, e))' \
 'input.query(e)=embed(e5, @query)' \
 'query=shipping stuff over the sea'
</pre>

<pre data-test="exec" data-test-assert-contains="id:doc:doc::1">
vespa query 'yql=select * from doc where userQuery() or ({targetHits: 100}nearestNeighbor(embedding, e))' \
 'input.query(e)=embed(e5, @query)' \
 'query=exchanging information by sound' 
</pre>

Remove the container after use:
<pre data-test="exec">
$ docker rm -f vespa
</pre>

## Ready for production

The E5-small-v2 [embedding model](https://huggingface.co/intfloat/e5-small-v2) used in this sample application
is suitable for production use and will produce good results in many domains without fine-tuning,
especially when combined with text match features.

## Model exporting
Transformer-based embedding models have named inputs and outputs that must  
be compatible with the input and output names used by the Vespa Bert embedder or the Huggingface embedder.


### Huggingface-embedder
See [export_hf_model_from_hf.py](export_hf_model_from_hf.py) for exporting a Huggingface sentence-transformer model to ONNX format compatible with default input and output names used by
the [Vespa huggingface-embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder). 

The following exports [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2):
<pre>
./export_hf_model_from_hf.py --hf_model intfloat/e5-small-v2 --output_dir model
</pre>


The following exports [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) using quantization:
<pre>
./export_hf_model_from_hf.py --hf_model intfloat/multilingual-e5-small --output_dir model --quantize
</pre>

The following exports [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) using quantization and tokenizer patching
to workaround this [issue](https://github.com/vespa-engine/vespa/issues/31086) with compatiblity problems with loading saved tokenizers:

<pre>
./export_hf_model_from_hf.py --hf_model intfloat/multilingual-e5-small --output_dir model --quantize --patch_tokenizer
</pre>


### Bert-embedder
Prefer using the [Vespa huggingface-embedder](https://docs.vespa.ai/en/embedding.html#huggingface-embedder) instead. 

See [export_model_from_hf.py](export_model_from_hf.py) for exporting a Huggingface sentence-transformer model to ONNX format compatible with default input and output names used by
the [bert-embedder](https://docs.vespa.ai/en/embedding.html#bert-embedder). 

The following exports [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) and saves the model parameters in an ONNX file and the `vocab.txt` file 
in the format expected by the Vespa bert-embedder.
<pre>
./export_model_from_hf.py --hf_model intfloat/e5-small-v2 --output_dir model
</pre>
