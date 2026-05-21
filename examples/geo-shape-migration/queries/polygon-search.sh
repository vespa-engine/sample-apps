#!/usr/bin/env bash
# Elasticsearch equivalent: geo_polygon { points: [...] }
#
# A user-drawn polygon (lat,lon pairs) is passed through the "polygon" query
# parameter. The PolygonSearcher injects a geoLocation() pre-filter from the
# polygon's bounding circle, then post-filters the result set with a
# ray-casting point-in-polygon test on each hit's "center" position.

# A triangle around Berlin center (Mitte / Potsdamer Platz / Friedrichshain).
POLYGON="52.55,13.30,52.55,13.45,52.49,13.40,52.55,13.30"

# Axis-aligned bounding box of the polygon — added to YQL as a content-side
# pre-filter so the Java polygon check in PolygonSearcher only sees candidates
# already inside the bbox. The YQL geoBoundingBox argument order is:
# (field, south, west, north, east).
SW_LAT=52.49
SW_LON=13.30
NE_LAT=52.55
NE_LON=13.45

curl -s -H "Content-Type: application/json" \
  --data "{
    \"yql\": \"select title, center from ad where geoBoundingBox(center, ${SW_LAT}, ${SW_LON}, ${NE_LAT}, ${NE_LON})\",
    \"polygon\": \"${POLYGON}\"
  }" \
  http://localhost:8080/search/ | jq '.root.children[]? | {title: .fields.title, center: .fields.center}'
