
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

<!-- ToDo: this is work in progress.
  This repo will keep code snippets for easy management of Vespa artifacts like log files in AWS
-->


# Vespa Code examples for AWS Lambda functions

Find deployable code examples in the `.py` files.
Make sure to start with the hello-example to make sure everything is set up,
using the AWS console Test function.

https://docs.aws.amazon.com/lambda/latest/dg/python-package.html has a good description
of how to create a package with dependencies - e.g.:
```
pip install --target ./package requests
cd package
zip -r ../my-deployment-package.zip .
cd ..
zip my-deployment-package.zip hello_function.py
```
You can have multiple python files in the same zip,
write one handler per python file.

Create an AWS Lambda function (replace 123456789012 with your account ID):
```
$ aws --profile private lambda create-function \
  --package-type Zip \
  --function-name hello_function \
  --zip-file fileb://my-deployment-package.zip \
  --runtime python3.9 \
  --handler hello_function.hello_handler \
  --role arn:aws:iam::123456789012:role/service-role/testf-role-bx2kyrtj
```
The `--profile` parameter can be used to deploy to a give AWS account - omit if having just one.
Refer to _.aws/config_ and _.aws/credentials_.

After a Lambda function is created, update it using:
```
$ aws --profile private lambda update-function-code \
  --function-name hello_function \
  --zip-file fileb://my-deployment-package.zip
```

Refer to [Google Cloud Functions](../../google-cloud/cloud-functions) for similar examples using Google Cloud.


## Troubleshooting
* Creating a Lambda in the AWS console with auto-create a role you can later use in the `--role` parameter.
* Using the AWS console to modify the code and rerun is quicker than re-deploying.
