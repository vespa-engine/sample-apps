
<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# Vespa sample applications - embedding service (WORK IN PROGRESS)

This sample application demonstrates how a Java [handler](https://docs.vespa.ai/en/jdisc/developing-request-handlers.html)
component can be used to process HTTP requests.
In this application, a handler is used to implement an embedding service,
which takes a string as an input and returns a vector embedding of that string.

## Setup

1. Follow steps 1-4 in the [Quick start, with Java](https://docs.vespa.ai/en/vespa-quick-start-java.html) guide.
2. Clone this repository: ``git clone INSERT_REPO_NAME && cd INSERT_REPO_NAME``.
3. Download the models:
```
cd src/main/application/
mkdir models && cd models
wget https://huggingface.co/intfloat/e5-small-v2
wget https://data.vespa.oath.cloud/onnx_models/e5-small-v2/tokenizer.json
cd ../../..
```
4. Follow steps 6 and 7 in the [Quick start, with Java](https://docs.vespa.ai/en/vespa-quick-start-java.html) guide to launch the application.

## Calling the embedding service

This sample application is a work in progress.
Currently, it has no GUI.
To interact with the application, you need to somehow send a POST request to the ``embedding`` endpoint,
containing a JSON object specifying the text to be encoded and the embedder to use.
Currently, only ``"hugging-face-embedder"`` is supported.

Here is a simple example using cURL:

    curl 'http://127.0.0.1:8080/embedding'  \
    -X POST --data-raw $'{ \
      "text": "text to embed", \
      "embedder": "hugging-face-embedder"  \
    }'

The output should look something like this:

    {
        "embedder":"hugging-face-embedder",
        "text":"text to embed",
        "embedding":"tensor<float>(x[384]):[-0.5786399, 0.20775521, ...]"
    }



