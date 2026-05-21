// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.geo;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.result.FeatureData;
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
 * lat/lon attributes place it outside the polygon, using a ray-casting test.
 *
 * The Searcher overrides the query's rank-profile to "searcher_geo", which
 * declares lat and lon as match-features. Match-features ship back with the
 * initial content-node response, so we read coordinates without triggering a
 * docsum fetch. fill() is never called from this Searcher — only the
 * surviving hits get filled (by the rendering layer) for whatever fields the
 * caller asked for in the YQL select clause.
 *
 * This requires Vespa 8.596.7+ (the version that skips docsum fetching when
 * only matchfeatures are used by the Searcher). See the inspiration:
 * https://vinted.engineering/2025/11/06/vespa-match-features/
 *
 * Combine this with a content-side pre-filter (geoBoundingBox(center, ...))
 * so the Searcher only sees candidates already inside the polygon's bbox.
 */
public class PolygonSearcher extends Searcher {

    private static final String POLYGON_PARAM = "polygon";
    private static final String MATCHFEATURES_FIELD = "matchfeatures";
    private static final String LAT_FEATURE = "lat_mf";
    private static final String LON_FEATURE = "lon_mf";
    private static final String SEARCHER_RANK_PROFILE = "searcher_geo";

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

        // Force the rank-profile that exposes lat/lon as match-features so we
        // can filter on coordinates without fetching summaries.
        query.getRanking().setProfile(SEARCHER_RANK_PROFILE);

        Result result = execution.search(query);
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

    /** Read lat/lon from the matchfeatures field that ships with the hit
     *  (no summary fetch needed). Vespa wraps matchfeatures as a FeatureData
     *  with a typed scalar accessor. */
    private double[] readPosition(Hit hit) {
        if (hit.getField(MATCHFEATURES_FIELD) instanceof FeatureData mf) {
            Double lat = mf.getDouble(LAT_FEATURE);
            Double lon = mf.getDouble(LON_FEATURE);
            if (lat != null && lon != null) {
                return new double[] { lat, lon };
            }
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
