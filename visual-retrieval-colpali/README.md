<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# 🔍 Visual Retrieval ColPali 👀

This readme contains the code for a web app including a frontend that can be set up as a user facing interface.

## Why?

To enable _you_ to showcase the power of ColPali and Vespa to your users, and to provide a starting point for your own projects.

> "But I only know Python, how can I create a web app that's not Gradio or Streamlit?" 🤔

No worries! This project uses [FastHTML](https://fastht.ml/) to create a beautiful web app - and it's all Python! 🐍

Also, 👇

<a href="https://imgflip.com/i/98mhch"><img src="https://i.imgflip.com/98mhch.jpg" title="made at imgflip.com" alt="Funny meme about json output in demo"/></a>

As a prerequisite, you should run [this notebook](https://vespa-engine.github.io/pyvespa/examples/visual_pdf_rag_with_vespa_colpali_cloud.html) 
to prepare the data and deploy the Vespa application. 

## Setting up your .env variables

The following variables are required in your `.env` file for the application to be able to connect to the Vespa application and the Gemini API:

You can rename the `.env.example` file to `.env` and fill in the required values.
The other variables are optional, if you want to use mTLS authentication against the Vespa application.

```bash
VESPA_APP_TOKEN_URL=https://abcde.z.vespa-app.cloud
VESPA_CLOUD_SECRET_TOKEN=vespa_cloud_xxxxxxxx
GEMINI_API_KEY=asdf
```

Optionally, set `GEMINI_MODEL` to use another Gemini model than the default `gemini-2.5-flash`.

If you want to deploy the application to Huggingface, you also need to set a `HF_TOKEN` variable, with write permissions.
This is personal, and must be created at [huggingface](https://huggingface.co/settings/tokens).

```bash
HF_TOKEN=hf_xxxxxxxxxx
```

## Setting up python environment

This application requires Python 3.10 or 3.11. The Hugging Face Space runs Python 3.11.

You can install the dependencies with `pip`, but we recommend using `uv`. 
Skip to [Installing dependencies using `uv`](#installing-dependencies-using-uv) if you want to use `uv`.

### Installing dependencies using `pip`

`src/requirements.txt` contains the pinned versions used by the Hugging Face Space:

```bash
pip install -r src/requirements.txt
```

### Installing dependencies using `uv`

We recommend installing the amazing `uv` to manage your python environment:
See also [installation - uv docs](https://docs.astral.sh/uv/getting-started/installation/) for other installation options.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, create a virtual environment:

```bash
uv venv 
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Sync your virtual environment with the dependencies:

```bash
uv sync --extra dev
```

## Running the application locally

To run the application locally, you can change into the `src` directory and run:

```bash
python main.py
```

This will start a local server, and you can access the application at `http://localhost:7860`.

## Deploy to huggingface 🤗 spaces

### Compiling dependencies

Before a deploy, make sure to run this to compile the dependencies to `requirements.txt` if you have made changes to the dependencies:

```bash
uv pip compile pyproject.toml --group huggingface --universal --python-version 3.11 -o src/requirements.txt
```

This will make sure that the dependencies in your `pyproject.toml` are compiled to the `src/requirements.txt` file, which is used by the huggingface space.
Run it from this directory, because `--group` reads the `pyproject.toml` in the current directory.
The `huggingface` group contains the Gradio SDK and `spaces` packages that Hugging Face installs next to `requirements.txt`, so the pins stay compatible with them.
If you change the Gradio version there, set the same `sdk_version` in `src/README.md`.

### Deploying to huggingface

Note that you need to set `HF_TOKEN` environment variable first.
This is personal, and must be created at [huggingface](https://huggingface.co/settings/tokens).
Make sure the token has `write` access.
Add the Vespa and Gemini variables from your `.env` as secrets in the Space settings.
Be aware that this will not delete existing files, only modify or add,
see [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/upload#upload-from-the-cli) for more
information.

#### Update your space configuration

The `src/README.md` file contains the configuration for the space.
Feel free to update this file to match your own configuration - name, description, etc.

Note that we can actually use the `gradio` SDK of spaces, to serve FastHTML apps as well, as long as we serve the app on port `7860`.
See [Custom python spaces](https://huggingface.co/docs/hub/en/spaces-sdks-python) for more information.

#### Upload the files

To deploy, run from the project root directory:

(Replace `vespa-engine/colpali-vespa-visual-retrieval` with your own huggingface user/repo name, does not need to exist beforehand)

```bash
hf upload vespa-engine/colpali-vespa-visual-retrieval src . --repo-type=space --delete "*" --exclude ".sesskey" --exclude "*__pycache__*" --exclude "static/full_images/**" --exclude "static/sim_maps/**" --commit-message "Deploy to HF Space"
```

Note that we upload only the `src` directory. The `--delete "*"` flag ensures that any files that exist in the space but not in your local `src` directory will be removed, keeping everything in sync.
The `--exclude` flags keep the local session key, Python caches and cached page images out of the Space.

## Testing and troubleshooting

Hugging Face builds the Space with a single `pip` command that installs `requirements.txt` together with the Gradio SDK from `sdk_version` and the `spaces` package.
To reproduce that environment and check that the app starts, run from this directory:

```bash
python3.11 -m venv /tmp/colpali-hf-test
/tmp/colpali-hf-test/bin/pip install -r src/requirements.txt "gradio[oauth,mcp]==6.29.0" "uvicorn>=0.14.0" "websockets>=10.4" spaces
/tmp/colpali-hf-test/bin/pip check
/tmp/colpali-hf-test/bin/python tests/huggingface_smoke.py --source src
```

The smoke test imports the app and requests its pages, with Vespa and the ColPali model mocked.
Add `--env-file .env --load-model` to also connect to your Vespa application and load the real model (about 6 GB on the first download).
The `Verify ColPali dependencies and startup` GitHub workflow runs these checks, including `--load-model`, on pull requests that change this directory.

- **The Space build fails at `pip install` with `ResolutionImpossible`:** a pin in `src/requirements.txt` conflicts with what Hugging Face installs next to it (for example, `gradio[mcp]` limits the `pydantic` version). Recompile `src/requirements.txt` as described above, and check that `sdk_version` matches the Gradio version in `pyproject.toml`.
- **Results look unrelated to the query, or the model load prints a `LOAD REPORT` with `MISSING` LoRA keys:** the installed `transformers` version doesn't load the ColPali adapter for this `colpali-engine` version. `--load-model` in the smoke test checks for this.
- **Searches return no results:** the `pdf_page` schema in your Vespa application is empty. Feed it with the notebook first.
- **`Please set the VESPA_APP_MTLS_URL environment variable`:** with `USE_MTLS=true`, the endpoint goes in `VESPA_APP_MTLS_URL`.
- **Static files are not found locally:** start the app from the `src` directory.

## Development

This section applies if you want to make changes to the web app.

### Adding dependencies

To add dependencies, you can add them to the `pyproject.toml` file and compile `src/requirements.txt` as described in [Compiling dependencies](#compiling-dependencies).

and then sync the dependencies:

```bash
uv sync --extra dev
```

### Making changes to CSS

To make changes to output.css apply, run

```bash
shad4fast watch # watches all files passed through the tailwind.config.js content section

shad4fast build # minifies the current output.css file to reduce bundle size in production.
```

### Instructions on creating and feeding the full dataset

This section is only relevant if you want to create and feed the full dataset to Vespa.
The notebook referenced in the beginning of this readme should be sufficient if you just want to spin up a scaled down version of the demo.

#### Prepare data and Vespa application

First, install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, run:

```bash
uv sync --extra dev --extra feed
```

#### Converting to notebook

If you want to run the `prepare_feed_deploy.py` as a notebook, you can convert it using `jupytext`:

Convert the `prepare_feed_deploy.py` to notebook to:

```bash
jupytext --to notebook prepare_feed_deploy.py
```

And launch a Jupyter instance with:

```bash
uv run --with jupyter jupyter lab
```

## Credits

Huge thanks to the amazing projects that made it a joy to create this demo 🙏🙌

- Freeing us from python dependency hell - [uv](https://astral.sh/uv/)
- Allowing us to build **beautiful** full stack web apps in Python [FastHTML](https://fastht.ml/)
- Introducing the ColPali architecture - [ColPali](https://huggingface.co/vidore/colpali-v1.2)
- Adding `shadcn` components to FastHTML - [Shad4Fast](https://github.com/curtis-allan/Shad4Fasthtml)
  
