<!-- Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

![Vespa logo](https://vespa.ai/assets/vespa-logo-color.png)

# access-log-lambda

This lambda function reads access log files from Vespa Documentation Search,
processes log lines into search suggestions by extracting query terms
and feeds the suggestions to the `query` schema.


### Clone repository

    $ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
    $ cd sample-apps/vespa-cloud/vespa-documentation-search/access-log-lambda/


### Install dependencies

    $ brew install nvm
    $ nvm install 14
    $ nvm use 14
    $ npm install


### Vespa private key for dataplane access

In AWS System Manager - Parameter Store, find a parameter named **VESPA_TEAM_DATA_PLANE_PRIVATE_KEY**
with the value of the private key to the Vespa Cloud application where queries should be fed.
Note that this secret is encrypted with a custom key -
this key must grant this lambda access for use.

Refer to <https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html>
for how to set up relevant keys.


### Set endpoint, public certificate and bucket name
In [index.js](index.js), set **vespaEndpoint** to the endpoint of the Vespa application where queries should be fed, set *publicCert* to the public certificate of the same Vespa application and set *Bucket* to the name of the bucket where the access logs are stored.


### Create lambda function
Create a lambda function named **access-log-lambda**


### Update role permissions
Give the lambda functions role the permissions **AmazonS3ReadOnlyAccess** and **AmazonSSMReadOnlyAccess**.
Also see https://console.vespa.oath.cloud/tenant/vespa-team/archive for more policies.


### Setup trigger
~~Setup a trigger on the S3 Bucket with access logs with the event type **ObjectCreatedByPut**.~~
Trigger once a day - info TBD.


### Configure lambda
For running the lambda with the example access log,
the lambda should be configured to use 256MB of memory and have an execution time of 30 seconds.
When running on all acces logs from one day,
the lambda should be configured to use 4096MB of memory and have an execution time of 30 seconds.


### Build Zip archive

    $ zip -r function.zip index.js node_modules/


### Deploy lambda
Upload the .zip-file from the AWS Lambda console - alternatively, deploy using `awscli`:

    $ brew install awscli
    $ aws configure
    $ aws lambda update-function-code --function-name access-log-lambda --zip-file fileb://function.zip
