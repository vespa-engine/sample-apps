<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Production Deployment with Java Tests

A minimal Vespa Cloud application for deployment into a Production zone - with basic Java-tests.

An application using Java test code must be deployed using the procedure for
[production deployment with components](https://cloud.vespa.ai/en/production-deployment#production-deployment-with-components) -
steps:

```
vespa config set target cloud
vespa config set application mytenant.myapp.myinstance
vespa auth login
mvn clean
vespa auth cert -f
mvn vespa:compileVersion -Dtenant=mytenant -Dapplication=myapp
mvn -U package -Dvespa.compile.version="$(cat target/vespa.compile.version)"
vespa prod deploy
```


## Developing system and staging tests
Develop tests using an instance in the Dev zone.
Use the Console and upload `target/application.zip` built in the steps above - use "default" instance name.

    mvn test -D test.categories=system \
             -D vespa.test.config=ext/test-config.json \
             -D dataPlaneCertificateFile=data-plane-public-cert.pem \
             -D dataPlaneKeyFile=data-plane-private-key.pem

    mvn test -D test.categories=staging-setup \
             -D vespa.test.config=ext/test-config.json \
             -D dataPlaneCertificateFile=data-plane-public-cert.pem \
             -D dataPlaneKeyFile=data-plane-private-key.pem

    mvn test -D test.categories=staging \
             -D vespa.test.config=ext/test-config.json \
             -D dataPlaneCertificateFile=data-plane-public-cert.pem \
             -D dataPlaneKeyFile=data-plane-private-key.pem


One can also use a local instance:

    mvn test -D test.categories=system        -D vespa.test.config=ext/test-config-local.json
    mvn test -D test.categories=staging-setup -D vespa.test.config=ext/test-config-local.json
    mvn test -D test.categories=staging       -D vespa.test.config=ext/test-config-local.json

See [Vespa Cloud Automated Deployments](https://cloud.vespa.ai/en/automated-deployments)
for an overview of production deployments.
