# access-log-lambda

## Steps

**Clone repository**

<pre>
$ git clone https://github.com/chunnoo/AccessLogLambda.git
$ cd AccessLogLambda
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
See <https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html> for description of how to set up relevant keys.


**Create parameter for vespa private key**

In AWS System Manager - Parameter Store, create a new parameter named **AccessLogPrivateKey** with the value of the private key to the Vespa application where queries should be fed.


**Set endpoint and public certificate**

In *index.js*, set **vespaHostname** to the endpoint of the Vespa application where queries should be fed and set *publicCert* to the public certificate of the same Vespa application.


**Create lambda function**

Create a lambda function named **access-log-lambda**


**Role permissions**

Give the lambda functions role the permissions **AmazonS3ReadOnlyAccess** and **AmazonSSMReadOnlyAccess**


**Setup trigger**

Setup a trigger on the S3 Bucket with access logs with the event type **ObjectCreatedByPut**


**Configure lambda**

For running the lambda with the example access log, the lambda should be configured to use 256MB of memory and have an execution time of 30 seconds. The execution time should probably be increased when running on production logs.


**Zip**

<pre>
$ zip -r function.zip index.js node_modules/
</pre>


**Deploy**

<pre>
$ aws lambda update-function-code --function-name access-log-lambda --zip-file fileb://function.zip
</pre>

Alternatively upload manually from the AWS Console.
