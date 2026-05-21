#!/usr/bin/env python3
"""Generate N synthetic ads scattered across Berlin (lat 52.4-52.6, lon 13.2-13.5).

Each doc has a center point, lat/lon attribute pair, and a small envelope.
Feed with:
    python3 bench/gen_synthetic_ads.py 50000 > /tmp/synthetic-ads.json
    vespa feed /tmp/synthetic-ads.json
"""
import json, random, sys

random.seed(42)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
LAT_MIN, LAT_MAX = 52.40, 52.60
LON_MIN, LON_MAX = 13.20, 13.50

print("[")
first = True
for i in range(N):
    lat = random.uniform(LAT_MIN, LAT_MAX)
    lon = random.uniform(LON_MIN, LON_MAX)
    if not first:
        print(",")
    first = False
    print(json.dumps({
        "put": f"id:ads:ad::syn-{i}",
        "fields": {
            "title": f"Synthetic ad {i}",
            "center": {"lat": lat, "lng": lon},
            "lat": lat, "lon": lon,
            "sw_lat": lat - 0.005, "ne_lat": lat + 0.005,
            "sw_lon": lon - 0.005, "ne_lon": lon + 0.005,
        }
    }))
print("]")
