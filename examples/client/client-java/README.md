# Usage

Make sure the top-level Vespa application is deployed.

Configure endpoint and identity files in `VespaClient.java`.

## Feed

```bash
gradle run --args="--feed ../dataset/documents-big.jsonl"
```

## Query

```bash
gradle run --args="--query"
```
