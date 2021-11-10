# Python-based text-image search app

Build a text-image search from scratch based on CLIP models with Vespa python API.

## Prepare your environment

### Download data

We are going to use the Flickr8k dataset to allow users to follow along from their laptop.

Download the image files (Flickr8k_Dataset.zip) from the address contained in this 
[file](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names) and
unzip the file.

### Install required packages

```
pip install -r requirements.txt
```

### Set environment variables

* Data: It is assumed that Flickr8k image files are locally stored on your disk.
```
export IMG_DIR='/path/to/image_folder/' # Folder with .png image files.
```

* Vespa deployment

In case you deploy to Vespa Cloud. See 
[this guide](https://pyvespa.readthedocs.io/en/latest/deploy-vespa-cloud.html) 
for more info about those variables.
```
export TENANT='your-tenant-name'
export APPLICATION='your-application-name'
export USER_KEY_PATH='your-user-key-path'
export DISK_FOLDER='path-to-hold-vespa-files'
```
In case you deploy locally to a Docker container:
```
export DISK_FOLDER='path-to-hold-vespa-files'
```

## Compare pre-trained CLIP models for text-image retrieval
> Create, deploy, feed and evaluate the Vespa app using the Vespa python API

### Jupyter Notebook

You can follow the example end-to-end by interactively running 
[this jupyter Notebook](https://github.com/vespa-engine/sample-apps/blob/master/text-image-search/src/python/compare-pre-trained-clip-for-text-image-search.ipynb).

### Demo the search app

Run a local demo of the text-image search app built here.

Set the following environment variables required by the app:
```
export VESPA_ENDPOINT = 'your-vespa-endpoint'
export VESPA_CERT_PATH= 'your-vespa-certificate-path'
export IMG_DIR='/path/to/image_folder/' # Folder with .png image files.
```
 
Run the app:
```
streamlit run app.py
```