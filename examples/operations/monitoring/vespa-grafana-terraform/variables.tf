variable "instance_url" {
  description = "URL to your Grafana instance"
  type        = string
}

variable "api_key" {
  description = "Grafana instance API key"
  type        = string
}

variable "prometheus_data_source_name" {
  description = "Name of Prometheus data source configured in Grafana"
  type = string
}

variable "alert_contact_point" {
  description = "Name of contact point for alert configuration"
  type = string
  default = null
}
