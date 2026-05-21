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
src/main/application/schemas/ad.sd        schema: center position + envelope + polygon rank-profile
src/main/application/services.xml         wires the PolygonSearcher into the chain
src/main/java/.../PolygonSearcher.java    point-in-polygon as a container Searcher (fallback)
src/test/application/tests/system-test/   HTTP-based system tests covering all patterns
feed/ads.json                             five-doc ad dataset
feed/polygon-test-points.json             five points from polygons.geojson (1, 5 inside; 2, 3, 4 outside)
queries/*.json                            executable query bodies — `vespa query --file …`
polygons.geojson                          non-convex 7-vertex polygon + 5 test points
```

## Run

```
mvn -U package
docker run --detach --name vespa --hostname vespa-container \
    --publish 8080:8080 --publish 19071:19071 \
    vespaengine/vespa
vespa config set target local
vespa status deploy --wait 300
vespa deploy --wait 300 target/application
vespa feed feed/ads.json
vespa feed feed/polygon-test-points.json
```

## Try the queries

Each pattern below ships as a JSON request body in `queries/`. Run with
`vespa query --file <file>`.

### 1. Service-area envelope contains point

```
vespa query --file queries/envelope-contains.json
```

Returns the three ads whose service-area envelope contains Berlin city
center (52.52, 13.405) — `berlin-mitte`, `berlin-wide`, and
`brandenburg-region` (whose bbox extends east into Berlin).

### 2. Envelope OR taxonomy path

```
vespa query --file queries/envelope-or-path.json
```

The Elasticsearch `bool.should = [terms location_path, geo_shape
contains point]` pattern — OR-combined directly in YQL. Returns the four
Berlin/Potsdam/Brandenburg ads tagged with `3331` or `3334`, plus any
extras whose envelope contains the search point.

### 3. Polygon search via container Searcher

```
vespa query --file queries/polygon-searcher.json
```

Runs the non-convex 7-vertex polygon from `polygons.geojson` through the
`PolygonSearcher` (ray-casting in Java, post-filter). Returns `pt-1`
and `pt-5`; drops `pt-2`, `pt-3`, `pt-4`.

### 4. Polygon search via content-node rank expression

```
vespa query --file queries/polygon-rankprofile.json
```

Same polygon, evaluated entirely on content nodes via the
`polygon_filter` rank-profile (ray-cast in tensor math; outside docs
dropped by `rank-score-drop-limit`). Same two hits — but the filter
runs *before* pagination, so `hits=N` returns exactly N.

## Tests

```
vespa test src/test/application/tests/system-test/geo-shape-test.json
mvn test     # unit-tests the polygon helpers
```

`vespa test` is idempotent — the first step deletes any leftover docs,
then re-feeds and re-runs every query above with assertions on hit
counts and titles.

---

## The three patterns

### 1. Service-area contains point — 4-field range query

Vespa has no `geo_shape` field type and no `relation: contains` operator.
The idiomatic equivalent is to **decompose the envelope into four scalar
attributes at index time** and let the query be four AND'ed range
conditions.

In `ad.sd`:

```
field sw_lat type double { indexing: attribute | summary; attribute: fast-search }
field ne_lat type double { indexing: attribute | summary; attribute: fast-search }
field sw_lon type double { indexing: attribute | summary; attribute: fast-search }
field ne_lon type double { indexing: attribute | summary; attribute: fast-search }
```

(No `rank: filter` here — see [below](#fast-search-and-rank-filter-on-the-bbox-fields--what-each-actually-does)
for why it would be a no-op on a high-cardinality `double` attribute.)

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
the `envelope-OR-path` step in `geo-shape-test.json`.

#### `fast-search` and `rank: filter` on the bbox fields — what each actually does

The two keywords are often pasted together as boilerplate. They have
very different runtime effects. For this app, only one of them is
load-bearing.

**`attribute: fast-search`** ([attributes guide][attr-doc],
[serving tuning][tune-doc]) builds an in-memory B-tree dictionary +
posting lists over the attribute. Without it, a query on the attribute
is a linear array scan — every range predicate visits every document's
value. With it, the same predicate is a B-tree lookup that yields the
matching docids in O(log N) per range endpoint, and the
`RangeQueryOptimizer` can fold several AND'ed range conditions on
fast-search attributes into a single `MultiRangeItem` pass.

- **When it pays off**: when the attribute can meaningfully reduce the
  candidate set on its own — i.e., the field is queried without
  another selective term, or is itself the primary filter. The
  serving-tuning guide says verbatim: *"Use fast-search if the
  attribute field is searched without any other query terms"* and
  *"if the attribute field could limit the total number of hits
  efficiently"*.
- **Cost**: extra in-memory dictionary + posting lists, persisted at
  steady state regardless of query traffic. Worse update throughput
  too, since the dictionary must be maintained.
- **For our four bbox fields**: the envelope-contains-point query is
  *only* the four range conditions — there is no co-occurring
  selective text filter to ride on. Without `fast-search` every
  envelope query would do four linear scans over 60M doubles.
  Memory overhead is a known and acceptable cost in exchange.

**`rank: filter`** ([schema reference][rank-doc]) is documented as
making the field "filter-only" — the matcher should skip collecting
ranking metadata. For *index* fields the effect is significant (skips
positions, switches to a Term-At-A-Time pass, drops per-doc ranking
data). For **single-value numeric attributes with `fast-search` it
does almost nothing in practice**. Walking the actual Vespa source:

1. `Ranking.setFilter(true)` (config-model) →
   `Attribute.setEnableOnlyBitVector(true)` (see
   [`AttributeFields.java:101`][af]).
2. The C++ config converter maps `enableonlybitvector=true` to
   `attribute.isFilter=true` (see
   [`configconverter.cpp:98`][cc]).
3. At query time this switches the iterator from `AttributeIteratorT`
   to `FilterAttributeIteratorT`. The only difference between their
   `doUnpack` implementations
   ([`attributeiterators.cpp`][ai]) is:
   ```cpp
   void AttributeIterator::doUnpack(uint32_t docId) {
       _matchData->resetOnlyDocId(docId);
       _matchPosition->setElementWeight(_weight);   // <-- only diff
   }
   void FilterAttributeIterator::doUnpack(uint32_t docId) {
       _matchData->resetOnlyDocId(docId);
   }
   ```
4. For a *single-value* numeric attribute, `_weight` is always 1. So
   `rank: filter` saves exactly one constant memory write per matched
   doc.
5. The other concrete effect is in `PostingStore`: once a posting list
   for one value grows large enough to become a bitvector, the
   underlying B-tree can be freed. That's a real memory saving — but
   only for fields with **repeating values** (categorical IDs, flags,
   coarse buckets). Our bbox corners are essentially unique floats; no
   single value will ever have a posting list large enough to trigger
   the bitvector replacement, so this optimization never fires here.

**For our bbox fields**, then, `rank: filter` would be decorative — it
documents intent but does not change query plans, memory footprint, or
measurable performance for the `select … where sw_lat <= X and
ne_lat >= X …` path. This app **omits** it on the four bbox fields for
that reason: less ceremony, identical behaviour.

`rank: filter` is kept on `location_path` because that's exactly the
case where it *does* earn its keep — a multi-value `array<string>`
attribute with repeating values, where the posting-list → bitvector
replacement actually fires and saves memory.

Other cases where `rank: filter` is load-bearing:
- **Categorical attributes with repeating values** (country codes,
  ACL bits, status flags): same bitvector replacement story as
  `location_path`.
- **Weighted-set attributes**: the skipped `setElementWeight` actually
  drops a meaningful per-element weight signal.
- **Index fields** (not attributes): skips position data and the
  Document-At-A-Time scoring path — substantial CPU win.

[attr-doc]: https://docs.vespa.ai/en/attributes.html
[tune-doc]: https://docs.vespa.ai/en/performance/feature-tuning.html
[rank-doc]: https://docs.vespa.ai/en/reference/schema-reference.html#rank
[af]: https://github.com/vespa-engine/vespa/blob/master/config-model/src/main/java/com/yahoo/schema/derived/AttributeFields.java
[cc]: https://github.com/vespa-engine/vespa/blob/master/searchlib/src/vespa/searchlib/attribute/configconverter.cpp
[ai]: https://github.com/vespa-engine/vespa/blob/master/searchlib/src/vespa/searchlib/attribute/attributeiterators.cpp

### 2a. Polygon search — bbox pre-filter + container post-filter (general fallback)

Vespa has no native point-in-polygon. The two-stage pattern works for any
polygon (convex or non-convex):

1. **Content-side pre-filter (in YQL)**: the client computes the polygon's
   axis-aligned bounding box and adds a
   `geoBoundingBox(center, sw_lat, sw_lon, ne_lat, ne_lon)` clause to the
   YQL. The content nodes evaluate this against the position attribute's
   Z-order curve, scaling to billions of documents.
2. **Container post-filter (in `PolygonSearcher`)**: the Searcher reads
   each hit's `center` from the summary and runs a ray-casting check.
   Hits outside the polygon are removed.

Activate by adding `"polygon": "lat1,lon1,lat2,lon2,..."` to any query
body; the trailing duplicate vertex (closed ring) is optional. See the
`polygon (Java Searcher)` step in `geo-shape-test.json` for the full
pattern, `PolygonSearcher.java` for the implementation, and
`PolygonSearcherTest.java` for unit tests on the geometry helpers.

### 2b. Polygon search — content-node rank expression (fast path)

The geometry test can also run **on the content nodes** as a first-phase
rank expression instead of in a container Searcher. Wins over 2a:

- Filtering happens before pagination — `hits=N` returns exactly N, no
  shrinkage from a post-filter dropping false positives.
- The polygon test is evaluated in parallel across content nodes/threads,
  not sequentially in one JVM.
- No round-trip parsing of the result hits to inspect each `center`.

The `polygon_filter` rank-profile implements ray-casting in tensor
math, so it handles **any simple polygon — convex or non-convex** — and
accepts arbitrary vertex winding.

**How it works**. The client passes the polygon as a tensor with one
entry per edge, where each entry holds the start and end coordinates of
that edge:

```
tensor<float>(edge{}, term{})   # term in {slat, slon, elat, elon}
```

For each edge, count whether a horizontal eastward ray from the query
point crosses it. The point is inside iff the total count is odd. The
"intersection east of point" test is rewritten as
`numerator * denominator > 0` to avoid division (and the NaN that would
come from horizontal edges).

In `ad.sd` (`polygon_filter` rank-profile, abridged):

```
function lat_brackets() {
    # 1 per edge if one endpoint is above the query lat and the other below.
    expression: map((query(polygon){term:slat} - attribute(lat))
                  * (query(polygon){term:elat} - attribute(lat)),
                    f(x)(if(x < 0, 1, 0)))
}
function east_of_point() {
    # 1 per edge if the ray-edge intersection is east of the query lon.
    expression: map( … numerator * denominator …, f(x)(if(x > 0, 1, 0)))
}
function crossings() {
    expression: reduce(lat_brackets * east_of_point, sum, edge)
}
function inside_polygon() {
    expression: if(fmod(crossings, 2) > 0.5, 1.0, 0.0)
}
first-phase {
    expression: if(inside_polygon > 0.5, nativeRank(title) + 1.0, -1.0)
    rank-score-drop-limit: 0.0
}
```

Outside points get score `-1.0`, which is below `rank-score-drop-limit:
0.0`, so they are dropped on the content nodes before pagination.

**Query**. Keep the `geoBoundingBox(center, …)` pre-filter so the rank
expression only fires on docs already inside the polygon's bbox; pass
the polygon vertices and select the rank-profile. The multi-mapped
tensor input uses the **nested-map JSON form**:

```json
{
  "yql": "select title, lat, lon from ad where geoBoundingBox(center, 52.49, 13.30, 52.55, 13.45)",
  "ranking": "polygon_filter",
  "input": {
    "query(polygon)": {
      "0": { "slat": 52.49, "slon": 13.30, "elat": 52.49, "elon": 13.45 },
      "1": { "slat": 52.49, "slon": 13.45, "elat": 52.55, "elon": 13.45 },
      "2": { "slat": 52.55, "slon": 13.45, "elat": 52.55, "elon": 13.30 },
      "3": { "slat": 52.55, "slon": 13.30, "elat": 52.49, "elon": 13.30 }
    }
  }
}
```

(Vespa rejects the flat form `{"0:slat": …}` for tensors with two
mapped dimensions — use the nested form.)

The `geojson polygon (rank-profile ray-cast …)` step in
`geo-shape-test.json` exercises this path against the 7-vertex
non-convex Berlin polygon in `polygons.geojson`.

---

## When to use which polygon path

Both 2a and 2b handle arbitrary simple polygons. The choice is about
the cost model — and the cost models are genuinely different.

### Measured numbers (50k synthetic Berlin docs, 7-vertex polygon)

Reproduce with:

```
python3 bench/gen_synthetic_ads.py 50000 > /tmp/synthetic-ads.json
vespa feed /tmp/synthetic-ads.json
python3 bench/bench_polygon.py
```

Numbers below are mean of 20 runs on an M-series Mac with a single-node
local Vespa container. Wall-clock includes HTTP; `backend_query` is the
content-node match+rank time, `backend_total` is `searchtime` from
`presentation.timing=true` (covers match + summary fetch + container
Searcher work).

Note on Vespa defaults: `maxHits = 400`. The "fair" Searcher rows below
override both `hits` and `maxHits` so the Searcher sees the same
candidate set the rank-profile evaluates on.

**Narrow query** (bbox == polygon's own bbox, ~600 docs pass the bbox
pre-filter):

| Path                                            | totalCount | backend_query | backend_total | wall   |
| ----------------------------------------------- | ---------- | ------------- | ------------- | ------ |
| rank-profile (content-node ray-cast)            | 173        | 3.6 ms        | 4.2 ms        | 6.7 ms |
| Java Searcher, hits=10 (top-N only)             | 4          | 0.5 ms        | 1.0 ms        | 2.3 ms |
| Java Searcher, hits=400 (Vespa default cap)     | 111        | 0.4 ms        | 1.8 ms        | 3.7 ms |
| Java Searcher, hits=1000 — **fair, sees all**   | 173        | 0.9 ms        | 2.6 ms        | 4.8 ms |

**Wide query** (bbox == all of Berlin, ~50k docs pass the bbox
pre-filter):

| Path                                            | totalCount | backend_query | backend_total | wall    |
| ----------------------------------------------- | ---------- | ------------- | ------------- | ------- |
| rank-profile (content-node ray-cast)            | 174        | 187 ms        | 188 ms        | 191 ms  |
| Java Searcher, hits=10 (top-N only)             | 2          | 1.8 ms        | 2.1 ms        | 3.8 ms  |
| Java Searcher, hits=400 (Vespa default cap)     | 4          | 1.8 ms        | 2.85 ms       | 4.5 ms  |
| Java Searcher, hits=60000 — **fair, sees all**  | 173        | 22 ms         | 263 ms        | 267 ms  |

### Napkin math (fair-comparison rows only)

**Rank-profile per-doc cost.** Wide-bbox query: ~187 ms over 50k
candidates ≈ **~3.7 µs per matched doc** for the 7-edge polygon ≈
~530 ns per edge. That's about two orders of magnitude above raw FP
throughput — the gap is tensor machinery (mapped-dim cell lookups,
broadcasting, `map` lambda dispatch). The rank-profile reads
`attribute(lat)` / `attribute(lon)` directly in the match phase and
**never fetches a summary** for outside-polygon docs.

**Java Searcher per-doc cost (fair).** The Searcher *must* call
`execution.fill(result)` to read `lat`/`lon`, so its cost includes
fetching the summary for every hit it sees:

- hits=60000 backend_total: ~263 ms ≈ **~5.3 µs per candidate**
- of which ~22 ms (≈0.4 µs/cand) is backend_query (match + heap
  collect of 60k hits)
- the remaining ~241 ms (≈4.8 µs/cand) is summary fetch + Java
  ray-cast + Hit object dereference

The actual Java ray-cast is sub-microsecond per hit; the dominant cost
is the **content→container summary fetch round-trip**, which the
rank-profile avoids entirely for outside-polygon docs.

### What the fair comparison actually shows

When forced to evaluate on the same K candidates:

| K       | rank-profile backend_total | Searcher backend_total | winner             |
| ------- | -------------------------- | ---------------------- | ------------------ |
| 600     | 4.2 ms                     | 2.6 ms                 | Searcher (~1.6× faster) |
| 50,000  | 188 ms                     | 263 ms                 | rank-profile (~1.4× faster) |

Crossover sits in the low thousands of candidates. Below that, the
Searcher's tight Java loop beats the tensor abstraction's per-cell
overhead. Above it, the Searcher's summary fetch cost overtakes the
in-place attribute read the rank-profile gets for free.

### Cost is not the same as correctness

Look back at the *unfair* Searcher rows (hits=10, hits=400) for the
wide query. They return `totalCount=4` or `totalCount=2` — not because
only 2–4 docs are inside the polygon (174 are), but because the backend
pre-ranks by `nativeRank(title)` (irrelevant to geometry), keeps the
top 400, and only those 400 are presented to the Searcher's
post-filter. The other ~170 inside-polygon docs are invisible to it.

So the production-relevant comparison is three-way:

| Question                                                | Best path                         |
| ------------------------------------------------------- | --------------------------------- |
| Cheapest if you only need the top-N relevant            | Java Searcher (cost bounded by hits) |
| Cheapest *correct* (matches totalCount, paginates ok)   | rank-profile (above ~1k candidates)  |
| Cheapest *correct* at small K (< ~1k candidates)        | Searcher with hits ≥ K (`maxHits` override) |
| Cheapest with very large polygons (>1k edges)           | Java Searcher (cost bounded by hits) |

### Summary

| Scenario                                            | Pattern                       |
| --------------------------------------------------- | ----------------------------- |
| Page-size correctness / accurate totalCount         | 2b — `polygon_filter` profile |
| Wide bbox ( ≳ 1k candidates ) + correctness         | 2b                            |
| Narrow bbox ( ≲ 1k candidates ) + correctness       | 2a — Searcher with `hits ≥ K` |
| Top-N relevant only, totalCount irrelevant          | 2a — Searcher with small hits |
| Polygon has thousands of edges                      | 2a (cost bounded by hits)     |

2b is the default choice in this app: it returns correct counts
without needing a `maxHits` override and scales sub-linearly in
practice with bbox cardinality. 2a is kept as a Java-side fallback
for very large polygons, narrow-bbox + correctness use cases (where
its tight Java loop beats the tensor overhead), and cases where
other container Searchers need to compose with the geometry filter.

---

## Answers to the five migration questions

**1. Is there a native "envelope contains query point"?**
No. Vespa's `position` field is a single point (or `array<position>`);
there is no shape type and no `geo_shape` operator. The 4-field range
query is the supported idiom and is what content nodes can index and
filter most efficiently.

**2. Is there a native "point in arbitrary polygon"?**
No. Two equivalent patterns are demonstrated here. Both handle convex
*and* non-convex polygons. The recommended choice is the
`polygon_filter` rank-profile (pattern 2b above), which evaluates the
ray-cast directly on content nodes via tensor math and drops
outside-polygon docs through `rank-score-drop-limit`. The container
Searcher (pattern 2a) is kept as a fallback for very large polygons
or when other container-side post-processing needs to compose with
the geometry filter.

**3. Where should the polygon test run — content node or container?**
Both are supported. The content-node path (rank expression +
`rank-score-drop-limit`) is faster, gives correct page sizes, and
parallelises across content nodes; it pays a small per-edge tensor
cost per matched document so very large polygons (thousands of edges)
can flip the trade-off. The container Searcher runs the ray-cast once
per result page in a single JVM, so its cost scales with page size
rather than candidate set size — better for huge polygons, worse for
the typical small-polygon mobile-map case.

**4. Can `geoLocation()` be used as a polygon approximation?**
Historically yes (a bounding circle of the polygon, with the polygon
test as a container post-filter). With pattern 2b you no longer need
this approximation — the content-side polygon test is exact and free
of false positives.

**5. Performance of 4-field range query at 60M docs?**
Good. The `RangeQueryOptimizer` (see link above) collapses the four
AND'ed `<= / >=` predicates on single-value attributes into one
`MultiRangeItem` that does a single pass over the fast-search btree;
the optimizer notes claim "query cost saving from this has been shown
to be 2 orders of magnitude in real cases". See [§1's
"`fast-search` and `rank: filter` on the bbox fields — what each
actually does"](#fast-search-and-rank-filter-on-the-bbox-fields--what-each-actually-does)
for what each of those keywords contributes (spoiler: `fast-search`
matters a lot, `rank: filter` is essentially decorative for a
high-cardinality `double`). One additional gotcha: range search on
`position` itself is *not* the way to do envelope-contains-point —
`position` is one point per doc, not the doc's bbox.

---

## Files of interest in the Vespa codebase

- [`RangeQueryOptimizer.java`](https://github.com/vespa-engine/vespa/blob/master/container-search/src/main/java/com/yahoo/search/querytransform/RangeQueryOptimizer.java) — merges AND'ed range conditions into `MultiRangeItem`.
- [`GeoLocationItem.java`](https://github.com/vespa-engine/vespa/blob/master/container-search/src/main/java/com/yahoo/prelude/query/GeoLocationItem.java) — query-tree item the Searcher injects.
- [`geo-search.html`](https://docs.vespa.ai/en/geo-search.html) — user guide for `geoLocation` and `geoBoundingBox`.
- [`yql.html#geoboundingbox`](https://docs.vespa.ai/en/reference/querying/yql.html#geoboundingbox) — YQL reference.
- [`schemas.html#position`](https://docs.vespa.ai/en/reference/schemas/schemas.html#position) — the `position` type.
- [`testing.html`](https://docs.vespa.ai/en/reference/applications/testing.html) — system-test JSON format used by `geo-shape-test.json`.
