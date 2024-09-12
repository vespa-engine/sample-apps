
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://assets.vespa.ai/logos/Vespa-logo-green-RGB.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg">
  <img alt="#Vespa" width="200" src="https://assets.vespa.ai/logos/Vespa-logo-dark-RGB.svg" style="margin-bottom: 25px;">
</picture>

# Grafana Terraform for Vespa Cloud

Terraform template for provisioning metric dashboards and alerts for your Vespa Cloud application. Supports both self-hosted Grafana and Grafana Cloud

## Prerequisites
* A running Grafana and Prometheus instance. See [album-recommendation-monitoring](https://github.com/vespa-engine/sample-apps/tree/master/examples/operations/monitoring/album-recommendation-monitoring/) for a sample setup

### Variables
* Grafana instance URL
* Grafana API key ([doc](https://grafana.com/docs/grafana/latest/administration/api-keys/#create-an-api-key))
* Name of Grafana datasource containing Vespa metrics. If there is no datasource yet, you can extend the template with a [grafana_data_source](https://grafana.com/docs/grafana-cloud/developer-resources/infrastructure-as-code/terraform/terraform-cloud-stack/#add-a-data-source)
* Alert contact point name (optional), determining the alert routing policy


### How to

In the base folder run
```
terraform init
terraform plan -var "api_key=${GRAFANA_API_KEY}" \
 --var "prometheus_data_source_name=${GRAFANA_DATASOURCE}" \
 --var "instance_url=${GRAFANA_INSTANCE_URL}" -out tf.plan
```
`tf.plan` can be inspected to see which changes will be applied. To apply them, run
```
terraform apply tf.plan
```
