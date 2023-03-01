terraform {
    required_providers {
        grafana = {
            source = "grafana/grafana"
            version = ">= 1.28.2"
        }
    }
}

resource "grafana_folder" "Vespa" {
   provider = grafana

   title = "Vespa metrics"
}

resource "grafana_dashboard" "vespa_dasbhoard" {
   provider = grafana

   for_each    = fileset("${path.module}/definitions", "*.tpl")
   config_json = templatefile("${path.module}/definitions/${each.key}", { datasource = "${var.prometheus_data_source_name}"})
   folder      = grafana_folder.Vespa.id
}
