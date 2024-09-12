
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications - embedding service (WORK IN PROGRESS)

This sample application demonstrates how a Java [handler](https://docs.vespa.ai/en/jdisc/developing-request-handlers.html)
component can be used to process HTTP requests.
In this application, a handler is used to implement an embedding service,
which takes a string as an input and returns a vector embedding of that string.

## Setup for Vespa Cloud deployment

### Cloud deployment

1. Create a new application in Vespa Cloud by following steps 1-4 in the [quick start guide](https://cloud.vespa.ai/en/getting-started)
2. Clone this repository: ``vespa clone examples/embedding-service embedding-service && cd embedding-service``
3. Download the models:
```
mkdir -p src/main/application/embedder-models/e5-small-v2
curl -o src/main/application/embedder-models/e5-small-v2/model.onnx \
  https://data.vespa-cloud.com/onnx_models/e5-small-v2/model.onnx
curl -o src/main/application/embedder-models/e5-small-v2/tokenizer.json \
  https://data.vespa-cloud.com/onnx_models/e5-small-v2/tokenizer.json
```
4. Add a public certificate: ``vespa auth cert``
5. Compile and deploy the application: ``mvn install && vespa deploy --wait 600``

### Enabling more embedders

By default, only the ``e5-small-v2`` embedder is enabled for cloud deployments.
Additional models are available, and can be enabled easily, though you should be mindful of the increased memory consumption.
Check out ``services.xml`` for more information.

## Setup for local deployment

1. Set up a Vespa Docker container by following steps 1-5 in the [quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html)
2. Clone this repository: ``vespa clone examples/embedding-service embedding-service && cd embedding-service``
3. Download the models:
```
mkdir -p src/main/application/embedder-models/e5-small-v2
curl -o src/main/application/embedder-models/e5-small-v2/model.onnx https://data.vespa-cloud.com/onnx_models/e5-small-v2/model.onnx
curl -o src/main/application/embedder-models/e5-small-v2/tokenizer.json https://data.vespa-cloud.com/onnx_models/e5-small-v2/tokenizer.json
```
4. Compile and deploy the application: ``mvn install && vespa deploy --wait 300``

### Adding more local embedders

More embedders from the [model hub](https://cloud.vespa.ai/en/model-hub) can be added
for local deployments, but this increases compile/deployment time.
To add a model, download its ``model.onnx`` and ``tokenizer.json`` files and add them
to a new subdirectory in ``src/main/application/embedder-models``.
Then, add it as a component in ``services.xml``.

## Calling the embedding service

This sample application is a work in progress.
Currently, it has no GUI.
To interact with the application, you need to somehow send a POST request to the ``embedding`` endpoint,
containing a JSON object specifying the text to be encoded and the embedder to use.

If you're using Vespa Cloud, you can use the ``vespa curl`` utility:

    vespa curl -- -X POST --data-raw \
    '{
        "text": "text to embed",
        "embedder": "e5-small-v2"
    }' \
    /embedding

If you're running the app locally, you can use normal ``curl``:

    curl 'http://127.0.0.1:8080/embedding'  \
    -X POST --data-raw  \
    '{ 
      "text": "text to embed", 
      "embedder": "e5-small-v2"  
    }'

The output should look something like this in both cases:

    {
        "embedder":"e5-small-v2",
        "text":"text to embed",
        "embedding":"tensor<float>(x[384]):[-0.5786399, 0.20775521, ...]"
    }



