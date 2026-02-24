# Usage

Make sure the top-level Vespa application is deployed.

Configure endpoint and identity files in `VespaClient.java`.

## Feed

```bash
gradle run --args="--feed ../dataset/docs.jsonl"
```

## Perform a simple query

```bash
gradle run --args="--query \"longest word in spanish\""
```

## Perform a query load test

```bash
gradle run --args="--load-test"
```
