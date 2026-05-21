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
field sw_lat type float { indexing: attribute | summary; attribute: fast-search }
field ne_lat type float { indexing: attribute | summary; attribute: fast-search }
field sw_lon type float { indexing: attribute | summary; attribute: fast-search }
field ne_lon type float { indexing: attribute | summary; attribute: fast-search }
```

**Type is `float`, not `double`.** At Berlin's latitude (≈52°) float's
quantum is **~0.7 m on lat, ~0.4 m on lon** — three orders of
magnitude finer than the underlying service-area data (km-scale
fuzz) and the query point (GPS ~5 m, usually rounded further by the
front-end). Halves the attribute footprint and the fast-search btree.
At 60M docs that's ~960 MB saved across the four fields with no
observable precision loss for envelope-contains-point. (`double`
would be appropriate if we were doing coord-system projection or
sub-meter precision arithmetic on stored values — we aren't.)

Note: changing an existing field's type requires a
`validation-overrides.xml` ack — see `src/main/application/validation-overrides.xml`.

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
   each hit's lat/lon from a `matchfeatures` payload that ships with
   the initial content-node response, runs a ray-casting check, and
   removes hits outside the polygon. **No `execution.fill(result)` is
   ever called from this Searcher** — only the surviving hits get
   their summary filled (by the rendering layer) for the fields the
   caller asked for in the YQL `select` clause.

Activate by adding `"polygon": "lat1,lon1,lat2,lon2,..."` to any query
body; the trailing duplicate vertex (closed ring) is optional. The
Searcher overrides the rank-profile to `searcher_geo` (which declares
`lat_mf` / `lon_mf` as match-features); see `ad.sd` for the profile
and `PolygonSearcher.java` for the read code.

#### Why this matters: skipping the docsum round-trip

Match-features were originally designed to expose ranking-feature
values for inspection / further ranking, but they have a second
property: they ride along with the initial content-node response, so
a container Searcher that only needs numeric values per hit can run
**before** any summary fetch and operate on the matchfeatures payload
alone. This is the trick described in
[Vinted Engineering's "Teaching the Old Dog a New Trick"][vinted];
it requires Vespa 8.596.7+, where Vespa correctly skips the docsum
fetch when the Searcher hasn't asked for one.

In numbers: at 50k bbox candidates, the naïve fill-based Searcher
spent ~240 ms fetching summaries before it could read each hit's
lat/lon. With match-features, that 240 ms vanishes; the Searcher gets
its data alongside the docids and only the (much smaller) post-filter
set gets summaries filled. See the [benchmark section
below](#measured-numbers-50k-synthetic-berlin-docs-7-vertex-polygon)
for the full numbers — the takeaway is that this collapsed a ~3×
gap with the rank-profile path down to ~1×.

`PolygonSearcherTest.java` unit-tests the geometry helpers; the
`geo-shape-test.json` system tests cover both query paths end-to-end.

[vinted]: https://vinted.engineering/2025/11/06/vespa-match-features/

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
| rank-profile (content-node ray-cast)            | 173        | 1.2 ms        | 1.9 ms        | 4.8 ms |
| Java Searcher, hits=10 (top-N only)             | 4          | 0.5 ms        | 1.0 ms        | 2.5 ms |
| Java Searcher, hits=400 (Vespa default cap)     | 111        | 1.1 ms        | 1.7 ms        | 3.8 ms |
| Java Searcher, hits=1000 — **fair, sees all**   | 173        | 1.1 ms        | 1.6 ms        | 4.3 ms |

**Wide query** (bbox == all of Berlin, ~50k docs pass the bbox
pre-filter):

| Path                                            | totalCount | backend_query | backend_total | wall    |
| ----------------------------------------------- | ---------- | ------------- | ------------- | ------- |
| rank-profile (content-node ray-cast)            | 174        | 13 ms         | 14 ms         | 17 ms   |
| Java Searcher, hits=10 (top-N only)             | 2          | 1.8 ms        | 2.1 ms        | 3.7 ms  |
| Java Searcher, hits=400 (Vespa default cap)     | 4          | 2.0 ms        | 2.4 ms        | 4.1 ms  |
| Java Searcher, hits=60000 — **fair, sees all**  | 173        | 193 ms        | 194 ms        | 198 ms  |

These numbers reflect the *optimized* `polygon_filter` profile, which
uses indexed-dim edge tensors and a single fused `join`. An earlier
version using mapped-dim tensors clocked ~190 ms backend_total on the
wide query — ~13× slower than the indexed-dim version. See
[Tensor-expression optimization journey](#tensor-expression-optimization-journey)
below for the before/after and what changed.

> The Java Searcher reads `lat` / `lon` from a `matchfeatures` payload
> that ships with the initial content-node response, so `fill()` is
> never called on the bbox-matched set — only on the much smaller
> post-filter survivor set. Prior to that change the same `hits=60000`
> row spent ~240 ms fetching summaries; see [§2a](#2a-polygon-search--bbox-pre-filter--container-post-filter-general-fallback)
> for the background.

### Tensor-expression optimization journey

The initial `polygon_filter` used a 2D mapped-dim tensor for the
polygon:

```
inputs {
    query(polygon) tensor<float>(edge{}, term{})
}
```

…with `term ∈ {slat, slon, elat, elon}` sliced out per edge as
`query(polygon){term:slat}` and so on, and two separate `map(…,
f(x)(if(x<0,1,0)))` passes to compute the lat-brackets / east-of-point
indicators before multiplying and reducing.

That version measured ~3.7 µs per matched doc (~530 ns per polygon
edge). Profiling the wide-bbox query with `vespa query --profile` /
`vespa inspect profile` showed first-phase at 203 ms over 50k
candidates — dominating end-to-end latency.

Three changes, applied together:

1. **Switch the `edge` dimension from mapped (`edge{}`) to indexed
   (`edge[32]`).** Mapped dimensions store cells in a hashmap and
   iterate via per-cell hash lookups. Indexed dimensions store cells
   in a contiguous array, which Vespa's tensor engine can stream
   through with vectorised (SIMD-friendly) kernels. The `term`
   dimension is gone — coordinates are passed as four separate
   tensors instead. Bound is 32 (covers any realistic polygon);
   trailing unused cells default to 0, which the cross-product test
   correctly treats as non-crossing.

2. **Fuse the two indicator `map`s into a single `join`.** The old
   code built two intermediate edge-tensors of 0/1 indicators, then
   multiplied them; the new code uses a 2-argument lambda inside one
   `join` to produce the indicator product in a single pointwise
   pass.

3. **Share `(elat - slat)` via a `dlat()` function** (the compiler
   may CSE this anyway; making it explicit is free if it does and a
   win if it doesn't).

Profile breakdown (`vespa inspect profile` on the wide query, 50k
bbox candidates):

| Phase           | mapped + 2× map (before) | indexed + join (after) | speedup     |
| --------------- | ------------------------ | ---------------------- | ----------- |
| matching        | 9.7 ms                   | 9.0 ms                 | (identical) |
| **first phase** | **203 ms**               | **30 ms**              | **6.9×**    |
| backend total   | 219 ms                   | 44 ms                  | 5.0×        |

Per-cell first-phase math:

- before: 203 ms / (50k docs × 7 edges) ≈ **580 ns / edge**
- after:  29.6 ms / (50k docs × 32 cells) ≈ **18 ns / edge-cell**

Even with 25 padded zero-cells per doc (32 cells worth of work for a
7-edge polygon), the per-cell cost is ~30× lower. The SIMD kernels
chew through the padding for almost free.

#### Profiling Vespa backends

Capture a profile and inspect it with:

```
vespa query --profile --file queries/polygon-rankprofile.json
vespa inspect profile
```

This drops `vespa_query_profile_result.json` in the working
directory and prints a table breaking the request into `matching`,
`first phase`, `second phase`, plus per-thread events. It's the
right tool for asking *"where is this query spending time on the
content nodes?"* — note that it does not see the container side
(so the Java Searcher's ray-cast doesn't show up here).

### Napkin math (fair-comparison rows only)

**Rank-profile per-doc cost.** Wide-bbox query: ~13 ms first-phase
over 50k candidates ≈ **~260 ns per matched doc** for the 7-edge
polygon, evaluated as 32 cells. The indexed-tensor join+reduce is
heavily SIMD-vectorised, so the actual arithmetic dominates over
framework overhead. The rank-profile reads `attribute(lat)` /
`attribute(lon)` directly in the match phase and drops
outside-polygon docs via `rank-score-drop-limit` before they reach
the container.

**Java Searcher per-doc cost (fair).** With match-features the
Searcher's cost decomposes differently. backend_query now includes
**match-feature computation** for every matched doc (the
`searcher_geo` rank-profile evaluates `lat_mf` and `lon_mf` per hit):

- hits=60000 backend_total: ~189 ms ≈ **~3.8 µs per candidate**
- backend_query ≈ 188 ms (match + heap collect of 60k + match-feature
  evaluation per matched doc)
- ~1 ms for `fill()` on the ~170 surviving hits
- summary fetch on the other ~49,830 outside-polygon hits: **skipped
  entirely** — they were filtered out by the Searcher before fill()
  was reached

So when the workloads are matched fairly, both paths cost roughly the
same per matched doc. The match-features round-trip carries
attribute data alongside docids; the rank-profile carries no data
out and finishes the filter in match phase.

### What the fair comparison actually shows

When forced to evaluate on the same K candidates:

| K       | rank-profile backend_total | Searcher backend_total | winner                       |
| ------- | -------------------------- | ---------------------- | ---------------------------- |
| 600     | 1.9 ms                     | 1.6 ms                 | Searcher (~1.2× faster, ~tied) |
| 50,000  | 14 ms                      | 194 ms                 | **rank-profile (~14× faster)** |

With the optimized indexed-dim rank-profile, the wide-K picture has
flipped: rank-profile is now far faster than the match-features
Searcher. The Searcher still has to ship matchfeatures for every
matched doc to the container and do its filtering in Java, paying
~3.9 µs/cand. The rank-profile completes in the match phase with
SIMD'd tensor kernels at ~260 ns/cand — roughly 15× lower per-doc
cost.

The earlier ~74 ms match-features optimization (vs the
summary-fetching baseline) is dwarfed by this 150 ms+ gain from the
tensor-expression rewrite. **The biggest performance lever was
choosing the right tensor type, not skipping the docsum fetch.**

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
| Wide bbox ( ≳ a few k candidates ) + correctness    | **2b** (~14× faster than 2a)  |
| Narrow bbox ( ≲ 1k candidates ) + correctness       | 2a or 2b (within ~20%)        |
| Top-N relevant only, totalCount irrelevant          | 2a — Searcher with small hits |
| Polygon has thousands of edges                      | 2a (bound `polygon_filter` to edge[N], or hits)  |

2b is the clear default for correctness-critical use cases at any
non-trivial candidate count. 2a remains useful when the cost is
bounded by `hits` (small page sizes, top-N retrieval) and the result
shrinkage from post-filtering is acceptable.

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
