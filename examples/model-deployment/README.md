<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# ONNX Model export and deployment example
Run the [ONNXModelExport](ONNXModelExport.ipynb) notebook to generate the model in the `models` directory.
To run the notebook, cd to this directory and run [jupyter](https://jupyter.org/install) -
this will open a browser window with the notebook:

    $ jupyter notebook

Or run in Colab: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vespa-engine/sample-apps/blob/master/examples/model-deployment/ONNXModelExport.ipynb)

Alternatively, run the script using python, or:

<pre data-test="exec" data-test-assert-contains="1 passed">
$ python3 -m pytest --nbmake examples/model-deployment/ONNXModelExport.ipynb
</pre>

Deploy the application package after completing the notebook.

Note: This example is work in progress and not completed yet
