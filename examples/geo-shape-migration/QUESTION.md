## Question 4: Geo Shape Contains + Geo Polygon
​
### Our ES Usage
​
**Geo Shape Contains** — checks if a document's stored service area (GeoJSON envelope) contains a search point:
​
```json
{
  "bool": {
    "should": [{
      "geo_bounding_box": {
        "geo_point": {
          "top_right": { "lat": 52.654, "lon": 13.627 },
          "bottom_left": { "lat": 52.385, "lon": 13.184 }
        }
      }
    }, {
      "geo_shape": {
        "geo_shape": {
          "shape": { "type": "point", "coordinates": [13.405, 52.52] },
          "relation": "contains"
        }
      }
    }]
  }
}
```
​
The `geo_shape` field on each document stores a GeoJSON envelope (bounding box of the ad's service area). The query asks: "does this ad's service area contain the user's search point?"
​
Combined with location_path as OR:
```json
{
  "bool": {
    "should": [{
      "terms": { "location_path": ["3331", "3334"] }
    }, {
      "geo_shape": {
        "geo_shape": {
          "shape": { "type": "point", "coordinates": [13.2653357, 52.5400284] },
          "relation": "contains"
        }
      }
    }]
  }
}
```
​
**Geo Polygon** — user-drawn polygon on mobile map:
​
```json
{
  "geo_polygon": {
    "geo_point": {
      "points": [
        { "lat": 52.5, "lon": 13.3 },
        { "lat": 52.6, "lon": 13.4 },
        { "lat": 52.4, "lon": 13.5 },
        { "lat": 52.5, "lon": 13.3 }
      ]
    }
  }
}
```
​
This is a closed polygon (first point == last point) for mobile map-view searches where users draw an area.
​
### How geo_shape is stored today in ES
​
The `geo_shape` field stores a GeoJSON **envelope** computed from the ad's center point + availability radius. At index time, we calculate 2 corners (northWest and southEast) and stores them as:
```json
"geo_shape": {
  "type": "envelope",
  "coordinates": [ [nw_lon, nw_lat], [se_lon, se_lat] ]
}
```
These 4 values (nw_lat, nw_lon, se_lat, se_lon) are only stored inside the single `geo_shape` field — there are **no separate numeric fields** for the bounding box corners today.
​
### Proposed Vespa Approach
​
**Geo Shape Contains:** Decompose the envelope into 4 separate numeric attribute fields at index time (`nw_lat`, `nw_lon`, `se_lat`, `se_lon`) and use range queries to check if a search point is contained:
```yql
where se_lat <= 52.52 AND nw_lat >= 52.52 AND nw_lon <= 13.405 AND se_lon >= 13.405
```
​
**Geo Polygon:** Use `geoBoundingBox()` as a fast pre-filter, then apply point-in-polygon evaluation in a custom Searcher.
​
### Questions
​
1. **Geo shape containment**: Is there any Vespa-native way to check if a document-stored bounding box contains a query point, other than the 4-field range query approach? Any planned support for spatial shape operations?
2. **Geo polygon filter**: Is there a native Vespa operator for "point within arbitrary polygon"? If not, what is the recommended approach — custom Searcher on content nodes, or container-level post-filter?
3. **Custom Searcher for polygon**: If we implement point-in-polygon in a custom Searcher, at which stage should it run — as a filter on content nodes (before ranking) or as a post-processing step in the container? What is the performance impact of a Java-based geometric check on every candidate document?
4. **`geoLocation()` as polygon approximation**: For convex polygons, could we use a bounding circle via `geoLocation()` as an over-select filter combined with a post-filter? What is the typical false-positive rate?
5. **The 4-field bounding box approach**: For the service-area containment, does storing `min_lat, max_lat, min_lon, max_lon` as separate attribute fields with range queries have performance concerns at 60M documents? Would a composite condition on 4 fields be optimized or evaluated sequentially?
​