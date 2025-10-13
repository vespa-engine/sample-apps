<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa Workshop - Quick Start Using Docker

See [https://docs.vespa.ai/en/vespa-quick-start.html](https://docs.vespa.ai/en/vespa-quick-start.html) for more information.

---

## 🐋 Download Docker

- **Linux:** [Install Docker Engine](https://docs.docker.com/engine/install/)
- **macOS:** [Install Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/)  
  or via Homebrew:
  ```bash
  brew install docker
  ```
- **Windows:** [Install Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)

---

## ⚙️ Install Vespa CLI

Vespa CLI is the official command-line client for Vespa.
Use it to deploy, feed, and query your Vespa applications locally or in the cloud.

### macOS / Linux

**Option 1: Install via Homebrew (macOS only)**  

```bash
brew install vespa-cli
```

**Option 2: Install manually using curl**

```bash
# macOS ARM64 (M1/M2/M3)
curl -LO https://github.com/vespa-engine/vespa/releases/download/v8.586.25/vespa-cli_8.586.25_darwin_arm64.tar.gz
tar -xzf vespa-cli_8.586.25_darwin_arm64.tar.gz
sudo mv vespa-cli_8.586.25_darwin_arm64/bin/vespa /usr/local/bin/vespa

# macOS Intel
curl -LO https://github.com/vespa-engine/vespa/releases/download/v8.586.25/vespa-cli_8.586.25_darwin_amd64.tar.gz
tar -xzf vespa-cli_8.586.25_darwin_amd64.tar.gz
sudo mv vespa-cli_8.586.25_darwin_amd64/bin/vespa /usr/local/bin/vespa

# Linux (amd64)
curl -LO https://github.com/vespa-engine/vespa/releases/download/v8.586.25/vespa-cli_8.586.25_linux_amd64.tar.gz
tar -xzf vespa-cli_8.586.25_linux_amd64.tar.gz
sudo mv vespa-cli_8.586.25_linux_amd64/bin/vespa /usr/local/bin/vespa
```

### Windows (PowerShell)

Download the windows release `.zip` from the [Vespa GitHub releases](https://github.com/vespa-engine/vespa/releases), extract it
and add the extracted folder (containing `bin/vespa.exe`) to your `PATH`.

See [Vespa CLI documentation](https://docs.vespa.ai/en/vespa-cli.html) for details.

---

## 🚀 Verify Prerequisites and Start Vespa

Verify that Docker and Vespa CLI are installed:
```bash
docker version
vespa version
```

Start a Vespa Docker container:
```bash
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
```

Deploy the Vespa application:
```bash
vespa config set target local
vespa deploy --wait 300 ./app
```

---

## 📦 Download Dataset

```bash
curl -O https://data.vespa-cloud.com/sample-apps-data/workshop/products.json
```

---

## 🔧 Convert Dataset to Vespa Feed Format

```bash
# On Linux / macOS:
./create_feed.py products.json products_feed.jsonl

# On Windows (PowerShell):
python create_feed.py products.json products_feed.jsonl
```

---

## 📤 Feed Data to Vespa

```bash
vespa feed products_feed.jsonl
```

---

## 🔍 Example Queries

```bash
vespa query \
  "yql=select * from product where name_description_index contains ({language: 'no'}'brød')" \
  "ranking.profile=native_rank" \
  "summary=debug-summary" \
  "presentation.timing=true" \
  "trace.level=7"

vespa query \
  "yql=select * from product where name_description_index_n_gram contains ({language: 'no'}'brød')" \
  "ranking.profile=native_rank_n_gram" \
  "summary=debug-summary" \
  "presentation.timing=true" \
  "trace.level=7"

vespa query \
  "yql=select * from product where name_description_attribute matches ({language: 'no'}'brød')" \
  "ranking.profile=native_rank_attribute" \
  "summary=debug-summary" \
  "presentation.timing=true" \
  "trace.level=7"
```
