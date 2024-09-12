
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Python-based text-image search app
This Python-based sample application uses [pyvespa](https://pyvespa.readthedocs.io/en/latest/index.html)
to process, feed and query Vespa. This is suitable for exploration and analysis.

## Download data

We are going to use the Flickr8k dataset to allow users to follow along from their laptop.
You can use the script in `../sh/download_flickr8k.sh` to download the data,
or manually from [the Kaggle website](https://www.kaggle.com/datasets/ming666/flicker8k-dataset).

After downloading, set the `IMG_DIR` environment variable to the folder containing the PNG files.

```
export IMG_DIR=<image-folder>
```

## Compare pre-trained CLIP models for text-image retrieval
Run [compare-pre-trained-clip-for-text-image-search.ipynb](https://github.com/vespa-engine/learntorank/blob/main/notebooks/compare-pre-trained-clip-for-text-image-search.ipynb)
for a full interactive end-to-end example that sets up a Vespa environment,
processes and feeds image data, and issues queries.
It includes an analysis over which of the six different pre-trained models in CLIP perform best.

This figure below shows the Reciprocal Rank @ 100 for each of the six
available pre-trained CLIP models.

-![alt text](../../resources/clip-evaluation-boxplot.png)


## Demo the search app
After running the notebook above, you can run the streamlit demo UI.
First, set the following environment variables required by the app:

```
export VESPA_ENDPOINT=<your-vespa-endpoint>
export VESPA_CERT_PATH=<your-vespa-certificate-path>  # if using Vespa Cloud, not needed for local Docker
export IMG_DIR=<image-folder>
```

Run the app:

```
streamlit run app.py
```

[Animation](https://data.vespa-cloud.com/sample-apps-data/image_demo.gif)
