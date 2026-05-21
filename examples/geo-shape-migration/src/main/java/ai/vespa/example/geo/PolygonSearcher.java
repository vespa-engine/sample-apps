// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.geo;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.Hit;
import com.yahoo.search.searchchain.Execution;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Implements "point inside arbitrary polygon" — Elasticsearch geo_polygon —
 * as a container-side post-filter.
 *
 * Triggered by a query parameter "polygon" with comma-separated lat,lon pairs
 * (open or closed ring). After the backend returns hits, drops any whose
 * "center" position lies outside the polygon, using a ray-casting test.
 *
 * Production guidance: combine this with a content-side pre-filter so the
 * Searcher only sees a narrow candidate set. The recommended pre-filter is to
 * AND a {@code geoBoundingBox(center, sw_lat, sw_lon, ne_lat, ne_lon)} clause
 * into the YQL itself, computed by the client from the polygon's axis-aligned
 * bounding box. {@code geoBoundingBox} is evaluated against the position
 * attribute's Z-order curve, which scales to billions of documents, so the
 * cost of the Java polygon check stays proportional to the result page size
 * rather than the corpus.
 */
public class PolygonSearcher extends Searcher {

    private static final String POLYGON_PARAM = "polygon";
    private static final String LAT_FIELD = "lat";
    private static final String LON_FIELD = "lon";

    @Override
    public Result search(Query query, Execution execution) {
        String polygonParam = query.properties().getString(POLYGON_PARAM);
        if (polygonParam == null || polygonParam.isBlank()) {
            return execution.search(query);
        }

        double[][] polygon = parsePolygon(polygonParam);
        if (polygon.length < 3) {
            query.trace("polygon needs at least 3 vertices, ignoring", 2);
            return execution.search(query);
        }

        // The YQL `select` clause restricts which fields the backend returns,
        // so callers using polygon= must include `lat` and `lon` in their
        // select list (or use `select *`). The geometry test below relies on
        // both being present.
        Result result = execution.search(query);
        execution.fill(result);
        int before = (int) result.hits().getConcreteSize();
        filterByPolygon(result, polygon);
        int after = (int) result.hits().getConcreteSize();
        query.trace("PolygonSearcher: " + before + " hits in -> " + after + " hits after polygon check", 2);
        return result;
    }

    private void filterByPolygon(Result result, double[][] polygon) {
        Iterator<Hit> it = result.hits().deepIterator();
        List<Hit> toRemove = new ArrayList<>();
        while (it.hasNext()) {
            Hit hit = it.next();
            if (hit.isMeta()) continue;
            double[] point = readPosition(hit);
            if (point == null || !pointInPolygon(point[0], point[1], polygon)) {
                toRemove.add(hit);
            }
        }
        for (Hit hit : toRemove) {
            result.hits().remove(hit.getId());
        }
        result.setTotalHitCount(result.hits().getConcreteSize());
    }

    private double[] readPosition(Hit hit) {
        Object lat = hit.getField(LAT_FIELD);
        Object lon = hit.getField(LON_FIELD);
        if (lat instanceof Number latN && lon instanceof Number lonN) {
            return new double[] { latN.doubleValue(), lonN.doubleValue() };
        }
        return null;
    }

    static double[][] parsePolygon(String s) {
        String[] parts = s.split(",");
        if (parts.length < 6 || parts.length % 2 != 0) return new double[0][];
        int n = parts.length / 2;
        double lat0 = Double.parseDouble(parts[0].trim());
        double lon0 = Double.parseDouble(parts[1].trim());
        double latN = Double.parseDouble(parts[parts.length - 2].trim());
        double lonN = Double.parseDouble(parts[parts.length - 1].trim());
        if (n > 1 && lat0 == latN && lon0 == lonN) n--;

        double[][] out = new double[n][2];
        for (int i = 0; i < n; i++) {
            out[i][0] = Double.parseDouble(parts[2 * i].trim());
            out[i][1] = Double.parseDouble(parts[2 * i + 1].trim());
        }
        return out;
    }

    /** Ray-casting test. Caller passes (lat, lon); polygon vertices are (lat, lon). */
    static boolean pointInPolygon(double lat, double lon, double[][] polygon) {
        boolean inside = false;
        for (int i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            double lat_i = polygon[i][0], lon_i = polygon[i][1];
            double lat_j = polygon[j][0], lon_j = polygon[j][1];
            boolean crosses = (lat_i > lat) != (lat_j > lat)
                              && lon < (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i) + lon_i;
            if (crosses) inside = !inside;
        }
        return inside;
    }
}
