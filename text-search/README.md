
# Data

Download data from [MS MARCO website](http://msmarco.org). Takes around 21G of space.

```bash
./bin/download-msmarco.sh
```

Extract, sample and convert data to Vespa JSON format.

```bash
./bin/convert-msmarco.sh
```

# Deploy the application package

In the following, we will assume you run all the commands from an empty
directory, i.e. the `pwd` directory is empty. We will map this directory
into the `/app` directory inside the Docker container. 

Now, to start the Vespa container:

```bash
docker run -m 12G --detach --name vespa-msmarco --hostname msmarco-app \
    --privileged --volume `pwd`:/app \
    --publish 8080:8080 --publish 19112:19112 vespaengine/vespa
```


Make sure that the configuration server is running - signified by a 200 OK response:

```bash
docker exec vespa-msmarco bash -c 'curl -s --head http://localhost:19071/ApplicationStatus'
```

To deploy theapplication:

```bash
docker exec vespa-msmarco bash -c '/opt/vespa/bin/vespa-deploy prepare /app/src/main/application && \
    /opt/vespa/bin/vespa-deploy activate'
```

After a short while, querying the port 8080 should return a 200 status code indicating that your application is up and running.

```bash
curl -s --head http://localhost:8080/ApplicationStatus
```

# Feed data

```bash
docker exec vespa-msmarco bash -c 'java -jar /opt/vespa/lib/jars/vespa-http-client-jar-with-dependencies.jar \
    --verbose --file /app/msmarco/vespa.json --host localhost --port 8080'
```

# Test query and evaluation scripts

Test query using `default` ranking profile:

```bash
curl -s http://localhost:8080/search/?query=is+sicily+a+state
```

Test query using `bm25` ranking profile:

```bash
curl -s http://localhost:8080/search/?query=is+sicily+a+state&ranking=bm25
```

The following two commands will generate two output files `test-output-default.tsv` 
and `test-output-bm25.tsv`, respectively. The files contain a reciprocal rank metric 
for each test query sent to Vespa. 

```bash
./src/python/evaluate.py default msmarco
./src/python/evaluate.py bm25 msmarco
```

The script `src/R/compare_rr.R` can be used to summarize those output files and to plot 
a boxplot comparing the performance of the two ranking functions.