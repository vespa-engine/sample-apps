#!/usr/bin/env bash
# Elasticsearch equivalent: bool.should = [ terms location_path, geo_shape contains point ]
# OR-combine the taxonomy filter with the envelope-contains-point check.

LAT=${1:-52.5400284}
LON=${2:-13.2653357}

curl -s -H "Content-Type: application/json" \
  --data "{
    \"yql\": \"select title, location_path, sw_lat, ne_lat, sw_lon, ne_lon from ad where location_path contains '3331' or location_path contains '3334' or (sw_lat <= ${LAT} and ne_lat >= ${LAT} and sw_lon <= ${LON} and ne_lon >= ${LON})\"
  }" \
  http://localhost:8080/search/ | jq '.root.children[] | {title: .fields.title, path: .fields.location_path}'
