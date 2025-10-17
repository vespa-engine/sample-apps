<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>


# Vespa sample applications - Job matching app with MCP server
This sample application combines a job search and job recommendation application with a built-in MCP server.
The point of this sample app is to provide a simple framework for you to experiment with and learn about Vespa's MCP capabilities.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/) or [Podman](https://podman.io/getting-started/installation)
- [Vespa CLI](https://docs.vespa.ai/en/vespa-cli.html)
- An MCP client (e.g. [Claude Desktop](https://claude.ai/download)

## Getting started
1. Clone this repository and navigate to the `mcp-server-app` directory:
```bash
git clone --depth 1 git@github.com:vespa-engine/sample-apps.git
cd sample-apps/examples/mcp-server-app
```
2. Start a Vespa container:
- With Docker:
```bash
docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
```
- With Podman:
```bash
podman run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
```
3. Deploy the application and feed data:
```bash
vespa config set target local
vespa application --wait 300
vespa feed ./dataset/*.jsonl --progress 2
```
4. Connect to the MCP server
- Using Claude Desktop:
Add 
```
"Vespa-mcp-server": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:8080/mcp/",
        "--transport",
        "http-first"
      ]
    }
```
to your `claude_desktop_config.json` file under the `McpServers` section.

## Mcp server capabilities
