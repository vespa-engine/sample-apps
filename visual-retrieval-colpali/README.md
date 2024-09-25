<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Visual Retrieval ColPali


# Developing

First, install `uv`:
  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, in this directory, run:

```bash
uv sync --extra dev
```

This will generate a virtual environment with the required dependencies at `.venv`.

To activate the virtual environment, run:

```bash
source .venv/bin/activate
```

And run development server:

```bash
python hello.py
```