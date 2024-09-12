
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# ONNX Model export and deployment example
Run the [ONNXModelExport](ONNXModelExport.ipynb) notebook to generate the model in the `models` directory.
To run the notebook, cd to this directory and run [jupyter](https://jupyter.org/install) -
this will open a browser window with the notebook:

    $ jupyter notebook

Or run in Colab: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vespa-engine/sample-apps/blob/master/examples/model-deployment/ONNXModelExport.ipynb)

Alternatively, run the script using python, or the pytest below, to generate the model.

Deploy the application package after completing the notebook.

Note: This example is work in progress and not completed yet


## Further reading
* [Ranking With ONNX Models](https://docs.vespa.ai/en/onnx.html)
