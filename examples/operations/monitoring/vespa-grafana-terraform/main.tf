terraform {
    required_providers {
        grafana = {
            source = "grafana/grafana"
            version = ">= 1.28.2"
        }
    }
}

provider "grafana" {
   url   = "${var.instance_url}"
   auth  = "${var.api_key}"
   cloud_api_key = "${var.api_key}"
}

module "dashboards" {
  source = "./modules/dashboards"
  prometheus_data_source_name = var.prometheus_data_source_name
}


module "alerts" {
  source = "./modules/alerts"
  prometheus_data_source_name = var.prometheus_data_source_name
  alert_contact_point = var.alert_contact_point
}
