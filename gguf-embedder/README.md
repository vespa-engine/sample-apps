# GGUF Embedder 

## Download Sample Data

```bash
wget -O- 'https://data.vespa-cloud.com/tests/performance/miracl-te-docs.10k.json.gz' | gunzip | jq '.[:100]' > miracl-te-docs.100.json
```

## Deploy the Application

```bash
vespa deploy
```