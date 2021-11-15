# Python-based text-image search app

Build a text-image search from scratch based on CLIP models with Vespa python API.

## Download data

We are going to use the Flickr8k dataset to allow users to follow along from 
their laptop. The data can be downloaded from [the Kaggle website](https://www.kaggle.com/ming666/flicker8k-dataset).

After downloading, set the `IMG_DIR` environment variable to the folder containing the PNG files.

```
export IMG_DIR=<image-folder>
```

## Compare pre-trained CLIP models for text-image retrieval
> Create, deploy, feed and evaluate the Vespa app using the Vespa python API

### Jupyter Notebook

You can follow the example end-to-end by interactively running
[this jupyter Notebook](https://github.com/vespa-engine/sample-apps/blob/master/text-image-search/src/python/compare-pre-trained-clip-for-text-image-search.ipynb).

### Demo the search app

![Text-Image Search with Vespa](../../resources/demo.gif)

Run a local demo of the text-image search app built here.

Set the following environment variables required by the app:
```
export VESPA_ENDPOINT = <your-vespa-endpoint>
export VESPA_CERT_PATH= <your-vespa-certificate-path> # if using Vespa Cloud
export IMG_DIR=<image-folder>
```

Run the app:
```
streamlit run app.py
```

