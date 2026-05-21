#!/usr/bin/env python3
"""Benchmark the two polygon paths against the synthetic Berlin corpus.

Compares:
  - Content-node rank-profile (`polygon_filter`, ray-cast in tensor math)
  - Container Java Searcher (`PolygonSearcher`, ray-cast in JVM)

Two query shapes:
  - NARROW: bbox == polygon's bbox (~600 candidates of a 50k corpus)
  - WIDE:   bbox == all-Berlin    (~50k candidates)

The same 7-vertex non-convex polygon from polygons.geojson is used in both
cases. The Java Searcher runs against three hits settings (10, 100, 400)
to expose how its cost scales with page size, not with candidate count.

Requires:
  1. Vespa running locally with the geo-shape-migration app deployed.
  2. Corpus fed via `python3 bench/gen_synthetic_ads.py 50000 | vespa feed -`
     (or the equivalent two-step pipe).
"""
import json, statistics, time, urllib.request

URL = "http://localhost:8080/search/"
N_WARMUP = 3
N_ITERS = 20
TIMEOUT_S = 60

POLY_LATLON = ("52.5354536,13.3799819,52.5316201,13.3633786,52.5291134,13.3763462,"
               "52.5134064,13.3933131,52.5346427,13.3974337,52.5250581,13.3871323,"
               "52.5325048,13.3796184")
POLY_TENSOR = {
    "0": {"slat": 52.5354536, "slon": 13.3799819, "elat": 52.5316201, "elon": 13.3633786},
    "1": {"slat": 52.5316201, "slon": 13.3633786, "elat": 52.5291134, "elon": 13.3763462},
    "2": {"slat": 52.5291134, "slon": 13.3763462, "elat": 52.5134064, "elon": 13.3933131},
    "3": {"slat": 52.5134064, "slon": 13.3933131, "elat": 52.5346427, "elon": 13.3974337},
    "4": {"slat": 52.5346427, "slon": 13.3974337, "elat": 52.5250581, "elon": 13.3871323},
    "5": {"slat": 52.5250581, "slon": 13.3871323, "elat": 52.5325048, "elon": 13.3796184},
    "6": {"slat": 52.5325048, "slon": 13.3796184, "elat": 52.5354536, "elon": 13.3799819},
}
NARROW_BBOX = (52.5134064, 13.3633786, 52.5354536, 13.3974337)  # polygon's bbox
WIDE_BBOX   = (52.4, 13.2, 52.6, 13.5)                         # all Berlin

def yql(bbox):
    return (f"select title, lat, lon from ad where "
            f"geoBoundingBox(center, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})")

def cases():
    """Two bbox shapes; for each, the rank-profile path plus three Searcher
    page sizes: hits=10 (typical), hits=400 (Vespa default cap), hits=K
    (page-size matching the bbox candidate count — the fair comparison)."""
    for label, bbox, K_full in [
        ("NARROW (bbox=polygon-bbox, ~600 candidates)", NARROW_BBOX, 1000),
        ("WIDE   (bbox=all-Berlin,   ~50k candidates)", WIDE_BBOX, 60000),
    ]:
        yield label, "rank-profile (content-node ray-cast, hits=400)", {
            "yql": yql(bbox), "ranking": "polygon_filter", "hits": 400,
            "input": {"query(polygon)": POLY_TENSOR},
            "presentation.timing": True, "timeout": f"{TIMEOUT_S}s",
        }
        for h in (10, 400):
            yield label, f"java Searcher, hits={h}", {
                "yql": yql(bbox), "polygon": POLY_LATLON, "hits": h,
                "presentation.timing": True, "timeout": f"{TIMEOUT_S}s",
            }
        yield label, f"java Searcher, hits={K_full} (fair: sees ALL bbox candidates)", {
            "yql": yql(bbox), "polygon": POLY_LATLON, "hits": K_full,
            "maxHits": K_full,
            "presentation.timing": True, "timeout": f"{TIMEOUT_S}s",
        }

def call(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read())
    return d, (time.perf_counter() - t0) * 1000

def stats(xs):
    xs = sorted(xs)
    return statistics.mean(xs), xs[len(xs)//2], xs[int(len(xs)*0.95)]

def run():
    prev_label = None
    for label, path_name, body in cases():
        if label != prev_label:
            print(f"\n=== {label} ===")
            prev_label = label
        for _ in range(N_WARMUP):
            call(body)
        wall, q, s = [], [], []
        for _ in range(N_ITERS):
            d, w = call(body)
            wall.append(w)
            t = d.get("timing", {})
            q.append(t.get("querytime", 0) * 1000)
            s.append(t.get("searchtime", 0) * 1000)
        total = d["root"]["fields"].get("totalCount", 0)
        wm, wp50, wp95 = stats(wall)
        qm, qp50, qp95 = stats(q)
        sm, sp50, sp95 = stats(s)
        print(f"  {path_name:42s}  totalCount={total:>5}")
        print(f"    wall_ms       mean={wm:6.2f}  p50={wp50:6.2f}  p95={wp95:6.2f}")
        print(f"    backend_query mean={qm:6.2f}  p50={qp50:6.2f}  p95={qp95:6.2f}")
        print(f"    backend_total mean={sm:6.2f}  p50={sp50:6.2f}  p95={sp95:6.2f}")

if __name__ == "__main__":
    run()
