<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

# Geo shape migration: Elasticsearch → Vespa

A minimal app that demonstrates Vespa equivalents for two Elasticsearch geo
queries that have no native counterpart in Vespa:

1. **`geo_shape` envelope contains point** — "does this document's stored
   service-area bounding box contain the user's location?"
2. **`geo_polygon`** — "is this document's location inside a user-drawn
   arbitrary polygon?"

The corpus is five ads, each with a center point, an optional
`location_path` taxonomy, and a stored service-area envelope. The dataset
is small on purpose so you can verify the expected hits by eye.

## Layout

```
src/main/application/schemas/ad.sd      schema with center position + envelope
src/main/application/services.xml       wires the PolygonSearcher into the chain
src/main/java/.../PolygonSearcher.java  point-in-polygon as a container Searcher
feed/ads.json                           five-doc dataset
queries/*.sh                            three example queries
```

## Run

```
mvn -U package
vespa deploy --wait 300 target/application
vespa feed feed/ads.json
queries/service-area-contains.sh                # default: Berlin city center
queries/service-area-or-path.sh                 # path-OR-envelope
queries/polygon-search.sh                       # triangle around Berlin
mvn test                                        # unit-tests the polygon helpers
```

---

## The two patterns

### 1. Service-area contains point — 4-field range query

Vespa has no `geo_shape` field type and no `relation: contains` operator.
The idiomatic equivalent is to **decompose the envelope into four scalar
attributes at index time** and let the query be four AND'ed range
conditions.

In `ad.sd`:

```
field sw_lat type double { indexing: attribute | summary; attribute: fast-search; rank: filter }
field ne_lat type double { indexing: attribute | summary; attribute: fast-search; rank: filter }
field sw_lon type double { indexing: attribute | summary; attribute: fast-search; rank: filter }
field ne_lon type double { indexing: attribute | summary; attribute: fast-search; rank: filter }
```

YQL for "does the document's envelope contain (52.52, 13.405)?":

```
where sw_lat <= 52.52 and ne_lat >= 52.52
  and sw_lon <= 13.405 and ne_lon >= 13.405
```

Why this scales: Vespa's
[`RangeQueryOptimizer`](https://github.com/vespa-engine/vespa/blob/master/container-search/src/main/java/com/yahoo/search/querytransform/RangeQueryOptimizer.java)
folds AND'ed range conditions over single-value attributes into a single
`MultiRangeItem` lookup that the backend evaluates against the btree
attribute index. `fast-search` is what enables that btree.

Convert the OR-combine-with-`terms` ES query (`bool.should = [terms
location_path, geo_shape contains point]`) by simply OR-ing in YQL — see
`queries/service-area-or-path.sh`.

### 2. Polygon search — bbox pre-filter + container post-filter

Vespa has no native point-in-polygon. The right pattern is two-stage:

1. **Content-side pre-filter (in YQL)**: the client computes the polygon's
   axis-aligned bounding box and adds a
   `geoBoundingBox(center, sw_lat, sw_lon, ne_lat, ne_lon)` clause to the
   YQL. The content nodes evaluate this against the position attribute's
   Z-order curve, scaling to billions of documents.
2. **Container post-filter (in `PolygonSearcher`)**: the Searcher reads
   each hit's `center` from the summary and runs a ray-casting check.
   Hits outside the polygon are removed.

Why post-filter in the container, not on the content nodes: there is no
content-side primitive for arbitrary polygons, and a Searcher runs once
per result page, not once per candidate document. With the bbox
pre-filter, the geometry test costs O(polygon_edges) per *hit*, not per
*doc in the corpus*.

Activate by adding `&polygon=lat1,lon1,lat2,lon2,...` to any query; the
trailing duplicate vertex (closed ring) is optional. See
`queries/polygon-search.sh` for the full pattern,
`PolygonSearcher.java` for the implementation, and
`PolygonSearcherTest.java` for unit tests on the geometry helpers.

---

## Answers to the five migration questions

**1. Is there a native "envelope contains query point"?**
No. Vespa's `position` field is a single point (or `array<position>`);
there is no shape type and no `geo_shape` operator. The 4-field range
query is the supported idiom and is what content nodes can index and
filter most efficiently.

**2. Is there a native "point in arbitrary polygon"?**
No. Recommended: pre-filter on `geoBoundingBox` (or `geoLocation` with
the polygon's bounding circle, as this app does), then run the polygon
test in a container Searcher.

**3. Where should the polygon test run — content node or container?**
Container. There is no supported way to plug arbitrary Java geometry
into the content node match phase, and you do not want to: keeping the
content side restricted to its indexed primitives (range, position,
nearest-neighbour) is what lets it scale to 60M+ documents. The Searcher
runs the polygon check on the already-filtered top hits, which is
proportional to `hits` (typically O(hundreds)) — fast enough that
ray-casting in Java is not a bottleneck. If you ever observe it
becoming one, narrow the pre-filter (tighter bounding circle / bbox) so
the Searcher sees fewer candidates.

**4. Can `geoLocation()` be used as a polygon approximation?**
Yes for convex polygons that are close to circular. The false-positive
rate is the area of (bounding circle − polygon) / area of bounding
circle, so it depends on the polygon's aspect ratio. For
mobile-map-view polygons that are roughly square, the over-select is
~30%, and dropping the false positives in the post-filter is cheap. For
long, thin polygons, prefer `geoBoundingBox` over the polygon's
axis-aligned bbox as the pre-filter — much tighter than a circle.

**5. Performance of 4-field range query at 60M docs?**
Good. The `RangeQueryOptimizer` (see link above) collapses the four
AND'ed `<= / >=` predicates on single-value attributes into one
`MultiRangeItem` that does a single pass over the fast-search btree;
the optimizer notes claim "query cost saving from this has been shown
to be 2 orders of magnitude in real cases". Three caveats:

- Use `attribute: fast-search` on all four bbox fields — without it,
  range queries fall back to linear attribute scans.
- Use `rank: filter` on bbox + `location_path` so they don't contribute
  to ranking metadata if you don't need them in scoring.
- Range search on `position` itself is *not* the way to do
  envelope-contains-point — `position` is one point per doc, not the
  doc's bbox.

---

## Files of interest in the Vespa codebase

- [`RangeQueryOptimizer.java`](https://github.com/vespa-engine/vespa/blob/master/container-search/src/main/java/com/yahoo/search/querytransform/RangeQueryOptimizer.java) — merges AND'ed range conditions into `MultiRangeItem`.
- [`GeoLocationItem.java`](https://github.com/vespa-engine/vespa/blob/master/container-search/src/main/java/com/yahoo/prelude/query/GeoLocationItem.java) — query-tree item the Searcher injects.
- [`geo-search.html`](https://docs.vespa.ai/en/geo-search.html) — user guide for `geoLocation` and `geoBoundingBox`.
- [`yql.html#geoboundingbox`](https://docs.vespa.ai/en/reference/querying/yql.html#geoboundingbox) — YQL reference.
- [`schemas.html#position`](https://docs.vespa.ai/en/reference/schemas/schemas.html#position) — the `position` type.
