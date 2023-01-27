<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

<!-- ToDo: this is work in progress.
  This repo will keep code snippets for easy management of Vespa artifacts like log files in Google Cloud
-->


# Vespa Code examples for Google Cloud functions

Find deployable code examples in [main.py](python/main.py) and [function.go](go/function.go).

The code examples include functions to list and parse Google Cloud Storage objects -
useful when parsing [access logs](https://docs.vespa.ai/en/access-logging.html).
Modify these to implement custom processing of these artifacts inside the Google Cloud zone.

Use `gcloud auth login` to log in with your credentials before deploying.

`gsutil` is a good utility to move objects to/from Google Cloud Storage, e.g.
`gsutil -o "GSUtil:parallel_process_count=1" -m cp -r access gs://mybucket_access_logs` -
This copies files into a Google Storage bucket.

Make sure to start with the hello-example to make sure everything is set up,
using `curl 'https://hello-abc5tvbrvq-ew.a.run.app/?name=Jim'`
(replace `abc5tvbrvq-ew` with your ID from the lambda deployment output).

Deploy a Google Function - python:
```
$ gcloud functions deploy getlogs \
  --gen2 \
  --region=europe-west1 \
  --runtime=python39 \
  --source=. \
  --entry-point=getlogs \
  --trigger-http
```
Note that the python file can have many functions, where you select which on to deploy above.

Deploy a Google Function - go:
```
$ gcloud functions deploy get-page-tls \
  --gen2 \
  --region=europe-west1 \
  --runtime=go119 \
  --source=. \
  --entry-point=getPageTLS \
  --trigger-http
```
Test using:
```
$ curl --data '{"url":"https://vespacloud-docsearch.vespa-team.aws-us-east-1c.z.vespa-app.cloud/document/v1/open/doc/docid"}' \
  https://get-page-tls-abc5tvbrvq-ew.a.run.app
```

Refer to [AWS Lambda Functions](../../aws/lambda) for similar examples using AWS Lambda.


## Troubleshooting
* Remember to enable Cloud Run API before deploying
* When deploying first time, you are asked if the function is runnable by anyone
* Functions can be invoked by `gcloud functions call hello --region europe-west1 --gen2 --data '{"name": "Jim"}'`
