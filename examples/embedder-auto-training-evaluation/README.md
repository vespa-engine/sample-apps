<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Automatic data generation for training embedders using LLMs

## What, how and why?

TLDR: Automatic data generation using the ChatGPT API in order to train an embedder to
perform better for information retrieval on specific datasets without labor-intensive and expensive manual training data annotation.

Machine learned embedder models enable efficient similarity computations,
but training these models requires large amounts of (often manually) annotated data.
The aim of this app is to investigate whether Large Language Models (LLMs), such as GPT-3.5-turbo,
can be employed to generate synthetic data for training embedder, without extensive manual intervention.

The repository contains scripts and notebooks to:
- Prepare datasets
- Generate training data for datasets using an LLM
- Train an embedder 
- Evaluate performance

## Overview of the process

### Query generation

1. Generate rules describing dataset
2. Use rules to create a new prompt for generating queries from documents
3. Use the prompt and example queries to get an LLM to generate one or more queries for a given document
4. Generate additional <a href="https://trec.nist.gov/data/qrels_eng/" data-proofer-ignore>qrels</a> (optional)
    1. Query Vespa using a generated query
    2. Ask an LLM to determine whether each returned document is relevant or not

### Training

1. Generate hard negatives for the generated queries
2. Train embedder
3. Evaluate with [trec_eval](https://github.com/usnistgov/trec_eval) and the nDCG@10 metric

## Setup 

Install the [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html):

```bash
brew install vespa-cli
```

Clone the repo:

```bash
vespa clone examples/embedder-auto-training-evaluation embedder-auto-training-evaluation
cd embedder-auto-training-evaluation
```

Install Python packages:

```bash
python3 -m pip install pyvespa ir_datasets transformers torch onnx sentence_transformers optimum[exporters]
```

Download an embedder model:

```bash
mkdir -p application-package/model
optimum-cli export onnx --framework pt --task sentence-similarity --model 'intfloat/e5-small-v2' application-package/model/
```

Download and prepare datasets using the [`ir_datasets`](https://ir-datasets.com/) package with the scripts in <a href="data-preparation/" data-proofer-ignore>data-preparation/</a>.
Scripts for the following [BEIR](https://github.com/beir-cellar/beir) datasets are available:

- FiQA
- NQ
- NFCorpus

Simply run **one** of the scripts with Bash:

```bash
bash data-preparation/prepare_fiqa.sh
bash data-preparation/prepare_nq.sh
bash data-preparation/prepare_nfcorpus.sh
```

### Setup Vespa

The repository contains an application package which sets
up an appropriate Vespa application. It can be deployed
either to Vespa Cloud or locally to Docker.
Feel free to change [application-package/services.xml](application-package/services.xml)
to change which resources should be available to the instance when deploying to Vespa Cloud.
We have found that using one or more GPU nodes massively speeds up feeding
(especially for NQ and other large datasets), as it embeds documents faster.

To create and deploy the app, first follow either steps 1-4 and 6 in the Vespa Cloud [getting started guide](https://cloud.vespa.ai/en/getting-started),
or steps 1-4 in the [Docker quick start guide](https://docs.vespa.ai/en/vespa-quick-start.html).
Then deploy the app, specifying the application package like this:

```bash
vespa deploy application-package --wait 600
```

Feed a dataset to the app by running **one** of the following commands,
depending on which dataset you're using:

```bash
vespa feed datasets/fiqa/feed.jsonl --progress 10
vespa feed datasets/nq/feed.jsonl --progress 10
vespa feed datasets/nfcorpus/feed.jsonl --progress 10
```

## Automatic query generation and training

When the app is deployed and running, and once you've fed your chosen dataset,
you can start generating queries and training models.

### Generate queries

Query generation is managed in the Jupyter notebook 
[generate_gpt_queries.ipynb](query-generation/generate_gpt_queries.ipynb).
There are plenty of options to play around with,
which can greatly affect the quality and cost of query generation.
The notebook contains more details.

Before proceeding, you need to create a file in the
<a href="query-generation/" data-proofer-ignore>query-generation/</a> directory called `.env`
containing your OpenAI API-key (generate one at https://platform.openai.com/account/api-keys).
This is required in order to use the OpenAI API to generate data.

After this is done, simply open up the notebook and follow the instructions there.

### Train and evaluate embedder

Training and evaluating an embedder is handled by a single script, [train_and_evaluate.sh](training-evaluation/train_and_evaluate.sh).
A GPU is recommended, but not required, to run the script.
The script first deploys the base model to Vespa, feeds documents, then generates hard negatives
for the generated queries.
Then the script trains a new model, deploys it, and evaluates the model on the provided test set.

To run the script you have to set an environment variable:

- `VESPA_ENDPOINT`, the URL to the search endpoint of your Vespa instance.

If deploying to Vespa Cloud you have to set two additional variables:

- `VESPA_KEY`, a path to a key for your Vespa Cloud instance.
- `VESPA_CERTIFICATE`, a path to a certificate for your Vespa Cloud instance.

The script can then be run with the following input paths:

- Output directory
- Documents file
- Queries to use for training
- Qrels to use for training
- Queries to use for testing
- Qrels to use for testing
