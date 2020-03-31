
## Prerequisites
```
pip3 install -r scripts/python_requirements.txt
```


## Download the CORD-19 dataset 
Download the dataset from [https://pages.semanticscholar.org/coronavirus-research](https://pages.semanticscholar.org/coronavirus-research):

Sample example uses the 2020-03-27 version:
```
mkdir -p data; cd data
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/comm_use_subset.tar.gz
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/noncomm_use_subset.tar.gz
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/custom_license.tar.gz
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/metadata.csv
for file in $(ls *tar.gz); do tar xzvf $file; done
```

## Download citation data from [scite_](https://scite.ai/)
See [blog post from scite](https://medium.com/scite/analyzing-more-than-1m-citations-to-better-understand-scientific-research-on-covid-19-3faa59d726c2) about this data.
Download data from [https://zenodo.org/record/3731542#.XoMiqdMzZBw](https://zenodo.org/record/3731542#.XoMiqdMzZBw)
```
wget "https://zenodo.org/record/3731542/files/covid-citations.csv?download=1" -o covid-citations.csv
wget "https://zenodo.org/record/3731542/files/covid-source-tallies.csv?download=1" -o covid-source-tallies.csv
cd ..
```

## Process the dataset
The optional set is for using sentence embedding representation for title and abstract using [SCIBERT-NLI](https://huggingface.co/gsarti/scibert-nli). 
```
python3 scripts/convert-to-json.py data/metadata.csv data/ 2020-03-27 > docs.json
#Optional python3 scripts/download-scibert-nli-model.py data/scibert-nli-model
#Optional python3 scripts/get-embeddings.py  docs.json data/scibert-nli-model > data/embeddings.csv 
python3 scripts/wash.py docs.json blacklist.txt > docs-washed.json
python3 scripts/compute-inbound-citations.py docs-washed.json > docs-washed-with-cited.json
python3 scripts/add-citation-data.py docs-washed-with-cited.json data/covid-source-tallies.csv data/covid-citations.csv > washed-with-citations.json
# Optional python3 scripts/add-embeddings.py washed-with-citations.json data/embeddings.csv > washed-with-citations-embeddings.json 
python3 scripts/convert-to-feed.py washed-with-citations.json > feed-file.json
```
Note that the input to the last process script needs to be changed if the optional add/embedding routine is used.

## Feed the data
Use [Vespa http feeding client](https://docs.vespa.ai/documentation/vespa-http-client.html) to feed the data to your Vespa instance
```
java -jar vespa-http-client-jar-with-dependencies.jar --file feed-file.json \
--endpoint <endpoint-url>  --verbose --useCompression
```






 





