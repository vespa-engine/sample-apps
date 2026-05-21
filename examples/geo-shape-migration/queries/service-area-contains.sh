#!/usr/bin/env bash
# Elasticsearch equivalent:
#   geo_shape: { shape: { type: point, coordinates: [13.405, 52.52] }, relation: contains }
#
# Find every ad whose service area envelope contains the user's location (Berlin
# city center, lat=52.52, lon=13.405). The four AND'ed range conditions are
# folded by Vespa's RangeQueryOptimizer into a single MultiRangeItem lookup on
# the fast-search btree attributes.

LAT=${1:-52.52}
LON=${2:-13.405}

curl -s -H "Content-Type: application/json" \
  --data "{
    \"yql\": \"select title, sw_lat, ne_lat, sw_lon, ne_lon from ad where sw_lat <= ${LAT} and ne_lat >= ${LAT} and sw_lon <= ${LON} and ne_lon >= ${LON}\"
  }" \
  http://localhost:8080/search/ | jq '.root.children[] | {title: .fields.title, bbox: [.fields.sw_lat, .fields.sw_lon, .fields.ne_lat, .fields.ne_lon]}'
