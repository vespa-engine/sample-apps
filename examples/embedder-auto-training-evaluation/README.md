# Vespa Summer Intern Project 2023

The fruits of our labor: Code, data, experiments and other cool stuff related to the summer intern project of 2023.

## What, how and why?

TLDR: Automagic data generation using the ChatGPT API in order to train an embedder model to
perform better on specific datasets without labor-intensive and expensive manual training data annotation.

Machine learned embedder models enable efficient similarity computations,
but training these models requires large amounts of (often manually) annotated data.
Our aim was to investigate whether Large Language Models (LLMs), such as GPT-3.5-turbo,
could be employed to generate synthetic data for training embedder models, without extensive manual intervention.

### Overview of the process

Query generation

1. Generate prompt-generating prompt
2. Use generated prompt to create new prompt
3. Use that prompt to generate query(ies)
4. Show it examples + a doc, then ask for a query
5. Consider the doc-query pair a valid qrel
6. Send the query to vespa and receive a list of hits
7. Show query and documents to LLM and ask for relevance judgement

Training

1. Train
2. Profit?

## How to use it

### Setup

#### Clone repo

```bash
git clone git@git.ouryahoo.com:tegge01/Vespa-Summer-Intern-Project-2023.git 
cd Vespa-Summer-Intern-Project-2023
```

#### Install tools and dependencies

1. Install Vespa CLI

```bash
brew install vespa-cli
```

2. Install Python packages

```bash
python3 -m pip install pyvespa ir_datasets transformers torch onnx
```

#### Download embedder model

```bash
mkdir -p application-package/model
python3 scripts/export_hf_model_from_hf.py --hf_model 'intfloat/e5-small-v2'  --output_dir application-package/model/
```

#### Download and prepare datasets

The scripts in [data-preparation](data-preparation) can be used to automatically download and process
various datasets using the **ir_datasets** package.
Scripts for the following datasets are available:

- FIQA
- NQ
- NFCORPUS

Simply run **one** of the scripts with bash:

```bash
bash data-preparation/prepare_fiqa.sh
bash data-preparation/prepare_nq.sh
bash data-preparation/prepare_nfcorpus.sh
```

#### Setup Vespa application

This repository contains an application package which sets
up an application in Vespa Cloud with additional GPU nodes.
Feel free to change [application-package/services.xml](application-package/services.xml)
to change which resources should be available to the instance.
We have found that using GPU nodes massively speeds up feeding
(especially for NQ and other large datasets), as it embeds documents faster.

##### Create and deploy app

Follow steps 1-4 and 6 in the Vespa Cloud [getting started guide](https://cloud.vespa.ai/en/getting-started).
Then deploy the app, specifying the application package like this:

```bash
vespa deploy application-package --wait 600
```

##### Feed app

Depending on which dataset you're using, run **one** of the following commands:

```bash
vespa feed datasets/fiqa --progress 10
vespa feed datasets/nq --progress 10
vespa feed datasets/nfcorpus --progress 10
```

### Generate queries

Query generation is managed in a Jupyter notebook called
[generate_gpt_queries.ipynb](query-generation/generate_gpt_queries.ipynb).
There are plenty of options to play around with,
which can greatly affect the quality and cost of query generation.
The notebook contains more details.

Before proceeding, you need to create a file in the [query-generation](query-generation) directory called **.env**
containing your OpenAI API-key (generate one at https://platform.openai.com/account/api-keys).
This is requires in order to use the OpenAI API to generate data.

After this is done, simply open up the notebook and follow the instructions there.

### Training and evaluating embedder

Training and evaluating an embedder is handled by a single script, [train_and_evaluate.sh](train_and_evaluate.sh).
It requires a GPU to run.
The script first deploys the base embedder to Vespa, feeds the dataset's documents, and generates hard negatives
for the train
and dev set.
Then the script trains a new model using those hard negatives, deploys it, and evaluates the model.

The script depends on another project. Download the repository with:
```bash
git clone git@github.com:Shybert/unilm.git
```

Install the script's required dependencies with:

```bash
python3 -m pip install -r unilm/simlm/requirements.txt
```

To run the script you have to set four environment variables:

- `DATA_DIR`, the path to a directory containing a prepared dataset.
- `VESPA_ENDPOINT`, the URL to the search endpoint of your Vespa instance.
- `VESPA_KEY`, a path to a key for your Vespa Cloud instance.
- `VESPA_CERTIFICATE`, a path to a certificate for your Vespa Cloud instance.

The script can then be run by passing the following inputs:

- The name of the output model
- The path to the queries of the train set
- The path to the qrels of the train set
- The path to the queries of the dev set
- The path to the qrels of the dev set
