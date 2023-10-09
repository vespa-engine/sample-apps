// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.data;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;

public class Promoted {

    static String query(int hits) {
        SimpleQueryBuilder query =  new SimpleQueryBuilder("/search/");
        query.add("yql", "select * from sources item where userQuery() and rating_count > 5");
        query.add("query", "sddocname:item");
        query.add("ranking", "promoted");
        query.add("summary", "default");
        query.add("hits", hits);
        return query.toString();
    }

    public static Map<String, JsonNode> data(SimpleHttpClient client, Map<String, String> properties) {
        int hits = Integer.parseInt(properties.getOrDefault("promoted", "10"));
        return Map.of("promoted", client.getJson(query(hits)));
    }

}
