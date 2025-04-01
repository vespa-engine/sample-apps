terraform {
    required_providers {
        grafana = {
            source = "grafana/grafana"
            version = ">= 1.28.2"
        }
    }
}

resource "grafana_notification_policy" "my_policy" {
    count = var.alert_contact_point != null ? 1 : 0
    group_by = ["alertname"]
    contact_point = "${var.alert_contact_point}"
}

resource "grafana_folder" "rule_folder" {
    title = "Vespa alerts"
}


data "grafana_data_source" "prometheus" {
  name                = "${var.prometheus_data_source_name}"
}

resource "grafana_rule_group" "vespa_rule_group" {
    name = "Vespa alert group"
    folder_uid = grafana_folder.rule_folder.uid
    interval_seconds = 60

    rule {
        name = "content-cluster-disk-util"
        annotations = {
          "runbook_url" = "https://docs.vespa.ai/en/operations/feed-block.html",
          "summary" = "A content node is experiencing high memory utilization"
        }
        condition = "C"
        for = "5m"

        data {
            ref_id = "A"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = data.grafana_data_source.prometheus.uid
            model = jsonencode({
                intervalMs = 1000
                maxDataPoints = 43200
                refId = "A"
                expr = "max by(applicationId, clusterId, zone) (content_proton_resource_usage_disk_average{zone=~\"prod.*\"})"
            })
        }

        data {
          ref_id = "B"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = "-100"
            model = file("${path.module}/reducer.json")
        }

        data {
            ref_id = "C"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = "-100"
            model = <<EOT
            {
          "conditions": [
            {
              "evaluator": {
                "params": [
                  0.9
                ],
                "type": "gt"
              },
              "operator": {
                "type": "and"
              },
              "query": {
                "params": [
                  "C"
                ]
              },
              "reducer": {
                "params": [],
                "type": "last"
              },
              "type": "query"
            }
          ],
          "datasource": {
            "type": "__expr__",
            "uid": "-100"
          },
          "expression": "B",
          "hide": false,
          "intervalMs": 1000,
          "maxDataPoints": 43200,
          "refId": "C",
          "type": "threshold"
        }
        EOT

        }
    
    }

    rule {
        name = "content-node-feed-blocked"
        annotations = {
          "runbook_url" = "https://docs.vespa.ai/en/operations/feed-block.html",
          "summary" = "A content node is feed blocked"
        }
        condition = "C"
        for = "5m"


        data {
            ref_id = "A"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = data.grafana_data_source.prometheus.uid
            model = jsonencode({
                intervalMs = 1000
                maxDataPoints = 43200
                refId = "A"
                expr = "content_proton_resource_usage_feeding_blocked_last{zone=~\"prod.*\"}"
            })
        }

        data {
          ref_id = "B"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = "-100"
            model = file("${path.module}/reducer.json")
        }

        data {
            ref_id = "C"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = "-100"
            model = <<EOT
            {
          "conditions": [
            {
              "evaluator": {
                "params": [
                  0
                ],
                "type": "gt"
              },
              "operator": {
                "type": "and"
              },
              "query": {
                "params": [
                  "C"
                ]
              },
              "reducer": {
                "params": [],
                "type": "last"
              },
              "type": "query"
            }
          ],
          "datasource": {
            "type": "__expr__",
            "uid": "-100"
          },
          "expression": "B",
          "hide": false,
          "intervalMs": 1000,
          "maxDataPoints": 43200,
          "refId": "C",
          "type": "threshold"
        }
        EOT

        }
    
    }

    rule {
        name = "content-cluster-mem-util"
        annotations = {
          "runbook_url" = "https://docs.vespa.ai/en/operations/feed-block.html",
          "summary" = "A content node is experiencing high memory utilization"
        }
        condition = "C"
        for = "5m"

        data {
            ref_id = "A"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = data.grafana_data_source.prometheus.uid
            model = jsonencode({
                intervalMs = 1000
                maxDataPoints = 43200
                refId = "A"
                expr = "max by(applicationId, zone, clusterId) (content_proton_resource_usage_memory_average{zone=~\"prod.*\"})"
            })
        }

        data {
          ref_id = "B"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = "-100"
            model = file("${path.module}/reducer.json")
        }

        data {
            ref_id = "C"
            relative_time_range {
                from = 600
                to = 0
            }
            datasource_uid = "-100"
            model = <<EOT
            {
          "conditions": [
            {
              "evaluator": {
                "params": [
                  0.8
                ],
                "type": "gt"
              },
              "operator": {
                "type": "and"
              },
              "query": {
                "params": [
                  "C"
                ]
              },
              "reducer": {
                "params": [],
                "type": "last"
              },
              "type": "query"
            }
          ],
          "datasource": {
            "type": "__expr__",
            "uid": "-100"
          },
          "expression": "B",
          "hide": false,
          "intervalMs": 1000,
          "maxDataPoints": 43200,
          "refId": "C",
          "type": "threshold"
        }
        EOT

        }
    
    }

    rule {
        name = "vespa-node-cpu-util"
        annotations = {
          "summary" = "A Vespa cluster has been experiencing high CPU for an extended period"
        }
        condition = "C"
        for = "30m"

        data {
            ref_id = "A"
            relative_time_range {
                from = 1800
                to = 0
            }
            datasource_uid = data.grafana_data_source.prometheus.uid
            model = jsonencode({
                intervalMs = 1000
                maxDataPoints = 43200
                refId = "A"
                expr = "max by(applicationId, zone, clusterId) (cpu_util{zone=~\"prod.*\"})"
            })
        }

        data {
          ref_id = "B"
            relative_time_range {
                from = 1800
                to = 0
            }
            datasource_uid = "-100"
            model = file("${path.module}/reducer.json")
        }

        data {
            ref_id = "C"
            relative_time_range {
                from = 1800
                to = 0
            }
            datasource_uid = "-100"
            model = <<EOT
            {
          "conditions": [
            {
              "evaluator": {
                "params": [
                  90
                ],
                "type": "gt"
              },
              "operator": {
                "type": "and"
              },
              "query": {
                "params": [
                  "C"
                ]
              },
              "reducer": {
                "params": [],
                "type": "last"
              },
              "type": "query"
            }
          ],
          "datasource": {
            "type": "__expr__",
            "uid": "-100"
          },
          "expression": "B",
          "hide": false,
          "intervalMs": 1000,
          "maxDataPoints": 43200,
          "refId": "C",
          "type": "threshold"
        }
        EOT

        }
    
    }

}