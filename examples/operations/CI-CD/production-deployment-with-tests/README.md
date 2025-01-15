<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Production Deployment with Tests

A minimal Vespa Cloud application for deployment into a Production zone - with basic tests. Steps:

```shell
vespa config set target cloud
vespa config set application mytenant.myapp.default
vespa auth login
vespa auth cert -f
vespa prod deploy
```

See [Production Deployment](https://cloud.vespa.ai/en/production-deployment) for details,
and read more about [Vespa Cloud Automated Deployments](https://cloud.vespa.ai/en/automated-deployments).
