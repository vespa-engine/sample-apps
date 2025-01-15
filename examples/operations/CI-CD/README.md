<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Vespa Cloud sample applications - Production Deployment with Tests

Vespa Cloud supports two kinds of CD tests:

* [production-deployment-with-tests](production-deployment-with-tests)
* [production-deployment-with-tests-java](production-deployment-with-tests-java)

![Illustration of system and staging tests](https://cloud.vespa.ai/assets/deployment-with-system-test.png)

See example GitHub workflow [deploy.yaml](.github/workflows/deploy.yaml).

Read more about
[Vespa Cloud Automated Deployments](https://cloud.vespa.ai/en/automated-deployments).
