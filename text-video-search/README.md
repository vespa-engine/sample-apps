
# Text-video search app

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

Build a text-video search from scratch based on CLIP models with Vespa python API.

[See Animation](https://data.vespa-cloud.com/sample-apps-data/video_demo.gif)

## Create the application in a Jupyter Notebook

Create, deploy, feed and query the application from [a single notebook](src/python/create-feed-query-text-video-search.ipynb)
with [Pyvespa](https://pyvespa.readthedocs.io/en/latest/index.html).


### Demo the search app

Run a local demo of the text-video search app built here.

Set the following environment variables required by the app:
```
export VESPA_ENDPOINT = <your-vespa-endpoint>
export VESPA_CERT_PATH= <your-vespa-certificate-path> # if using Vespa Cloud
export VIDEO_DIR=<video-folder>
```

Run the app:
```
streamlit run src/python/app.py
```
