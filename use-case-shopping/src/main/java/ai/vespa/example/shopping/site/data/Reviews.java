// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.data;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;

public class Reviews {

    static String query(Map<String, String> properties) {
        String order = "timestamp";
        String orderDir = "desc";
        if (properties.containsKey("sortreview")) {
            switch (properties.get("sortreview")) {
                case "tn": order = "timestamp"; orderDir = "desc"; break;
                case "to": order = "timestamp"; orderDir = "asc"; break;
                case "hr": order = "stars"; orderDir = "desc"; break;
                case "lr": order = "stars"; orderDir = "asc"; break;
                case "mh": order = "upvotes"; orderDir = "desc"; break;
                case "lh": order = "upvotes"; orderDir = "asc"; break;
            }
        }

        String select = "select * from sources review where ";
        String where = "asin contains \"" + properties.get("i") + "\" ";
        String orderBy = "order by " + order + " " + orderDir;
        String grouping = " all( group(stars) each(output(count())) )";

        SimpleQueryBuilder query = new SimpleQueryBuilder("/search/");
        query.add("yql", select + where + orderBy + " | " + grouping);
        query.add("summary", "default");
        query.add("hits", 400);

        return query.toString();
    }

    public static Map<String, JsonNode> data(SimpleHttpClient client, Map<String, String> properties) {
        return Map.of("reviews", client.getJson(query(properties)));
    }

}
