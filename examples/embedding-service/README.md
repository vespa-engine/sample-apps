
<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - embedding service (WORK IN PROGRESS)

This sample application demonstrates how a Java [handler](https://docs.vespa.ai/en/jdisc/developing-request-handlers.html)
component can be used to process HTTP requests.
In this application, a handler is used to implement an embedding service,
which takes a string as an input and returns a vector embedding of that string.

## TODO Setup for Vespa Cloud deployment

## Setup for local deployment

1. Set up a Vespa Docker container by following steps 1-5 in the [quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html)
2. Clone this repository: ``vespa clone examples/embedding-service embedding-service``
3. Download the models:
```
mkdir -p src/main/application/embedder-models/e5-small-v2
wget -P src/main/application/embedder-models/e5-small-v2 https://data.vespa.oath.cloud/onnx_models/e5-small-v2/model.onnx
wget -P src/main/application/embedder-models/e5-small-v2 https://data.vespa.oath.cloud/onnx_models/e5-small-v2/tokenizer.json
```
4. Compile and deploy the application: ``mvn install && vespa deploy --wait 300``

## Calling the embedding service

This sample application is a work in progress.
Currently, it has no GUI.
To interact with the application, you need to somehow send a POST request to the ``embedding`` endpoint,
containing a JSON object specifying the text to be encoded and the embedder to use.

Currently, only ``"e5-small-v2"`` is supported for local deployments.
If you're running the app in Vespa Cloud,
``"e5-base-v2"``, ``"e5-large-v2"``, ``"multilingual-e5-base"``and ``"minilm-l6-v2"``
are also available.


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

## Adding more local embedders

More embedders from the [model hub](https://cloud.vespa.ai/en/model-hub) can be added
for local deployments, but this increases compile/deployment time.
To add a model, download its ``model.onnx`` and ``tokenizer.json`` files and add them
to a new subdirectory in ``src/main/application/embedder-models``.
Then, add it as a component in ``services.xml``.


