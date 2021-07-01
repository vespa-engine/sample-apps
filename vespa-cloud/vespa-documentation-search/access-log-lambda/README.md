# access-log-lambda

## Steps

**Clone repository**

<pre>
$ git clone --depth 1 https://github.com/vespa-engine/sample-apps.git
$ cd sample-apps/vespa-cloud/vespa-documentation-search/access-log-lambda/
</pre>


**Dependencies**

<pre>
$ brew install awscli
$ brew install nvm
$ nvm install 14
$ nvm use 14
$ npm install
</pre>


**Configure AWS**

<pre>
$ aws configure
</pre>
See <https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html>
for description of how to set up relevant keys.


**Create parameter for vespa private key**

In AWS System Manager - Parameter Store, find a parameter named **VESPA_TEAM_DATA_PLANE_PRIVATE_KEY**
with the value of the private key to the Vespa Cloud application where queries should be fed.

Note that this secret is encrypted with a custom key -
this key must grant this lambda access for use.


**Set endpoint and public certificate**

In [index.js](index.js), set **vespaEndpoint** to the endpoint of the Vespa application where queries should be fed
and set *publicCert* to the public certificate of the same Vespa application.


**Create lambda function**

Create a lambda function named **access-log-lambda**


**Role permissions**

Give the lambda functions role the permissions **AmazonS3ReadOnlyAccess** and **AmazonSSMReadOnlyAccess**.
Also see https://console.vespa.oath.cloud/tenant/vespa-team/archive for more policies.


**Setup trigger**

Setup a trigger on the S3 Bucket with access logs with the event type **ObjectCreatedByPut**.
Alternatively, trigger time-based


**Configure lambda**

For running the lambda with the example access log,
the lambda should be configured to use 256MB of memory and have an execution time of 30 seconds.
The execution time should probably be increased when running on more log files.


**Zip**

<pre>
$ zip -r function.zip index.js node_modules/
</pre>


**Deploy**

<pre>
$ aws lambda update-function-code --function-name access-log-lambda --zip-file fileb://function.zip
</pre>

Alternatively upload manually from the AWS Console.
