<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>


# Vespa sample applications - Job matching app with MCP server
This sample application combines a job search and job recommendation application with a built-in Model Context Protocol (MCP) server.
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
<pre data-test="exec">
docker pull vespaengine/vespa
docker run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
<pre/>
- With Podman:
```bash
podman pull vespaengine/vespa
podman run --detach --name vespa --hostname vespa-container \
  --publish 127.0.0.1:8080:8080 --publish 127.0.0.1:19071:19071 \
  vespaengine/vespa
```
3. Deploy the application and feed data:
<pre data-test="exec" data-test-assert-contains="Success">
vespa config set target local
vespa deploy application --wait 300
vespa feed ./dataset/*.jsonl --progress 2
</pre>

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
### Tools
- **`executeQuery`**: Build and execute Vespa queries against the Vespa application.
- **`getSchemas`**: Retrieve the schemas of the Vespa application.
- **`searchDocumentation`**: Search the [Vespa documentation](https://docs.vespa.ai/) for relevant information based on a user input.

### Resources
- **`queryExamples`**: Provides query examples to the MCP client for guidance on how to use the `executeQuery` tool.

### Prompts
- **`listTools`**: Prompt to list the tools and their descriptions of the MCP server.

## App exploration
Since the point of the sample app is to become familiar with Vespa's MCP server capabilities, here are some tasks and questions to explore:
- Find a random candidate amongst your documents and try to find the best matching job for this candidate.
- Of the jobs matching this candidate, where would our candidate have the best chances to land a job? Do any of the jobs have other better candidates?
- Based on your skills and interests, do any of the jobs match your profile?
- Ask the LLM other Vespa related questions and have it search the documentation for you.
- Can the application be improved? Maybe the LLM can help you modify the schemas and datasets? Make sure `generate_data.py` actually generates data that matches the schemas if you modify them.

### Generating data
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
```