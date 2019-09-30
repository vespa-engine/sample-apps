// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.text_search.site.data;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;
import java.util.StringJoiner;

public class SearchResults {

    static String query(Map<String, String> properties) {
        SimpleQueryBuilder query = new SimpleQueryBuilder("/search/");

        String q = properties.get("q");
        String yql = "select * from sources msmarco where userInput(\"" + q + "\");";

        query.add("yql", yql);
        query.add("summary", properties.getOrDefault("s", "default"));
        query.add("ranking", properties.getOrDefault("r", "bm25"));
        query.add("hits", properties.getOrDefault("h", "10"));
        query.add("ranking", properties.getOrDefault("profile", "bm25"));

        if (properties.containsKey("p")) {
            int pageVal = Integer.parseInt(properties.get("p"));
            int perPage = Integer.parseInt(properties.getOrDefault("h", "10"));
            query.add("offset", pageVal * perPage);
        }

        return query.toString();
    }

    public static Map<String, JsonNode> data(SimpleHttpClient client, Map<String, String> properties) {
        return Map.of("searchresults", client.getJson(query(properties)));
    }

}
