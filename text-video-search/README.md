# Text-video search app

Build a text-video search from scratch based on CLIP models with Vespa python API.

![Text-Video Search with Vespa](demo.gif)

## Create the application from scratch in a Jupyter Notebook

Create, deploy, feed and query the application from [a single notebook](src/python/create-feed-query-text-video-search.ipynb)
with [the Vespa python API](https://pyvespa.readthedocs.io/en/latest/index.html).

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

