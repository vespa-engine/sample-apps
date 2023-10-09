// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.data;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;

public class Categories {

    static String query() {
        String yql = "select * from sources item where true";
        String grouping = "all( " +
                "group(array.at(categories,0)) order(-count()) each( " +
                    "output(count()) all(" +
                        "group(array.at(categories,1)) order(-count()) each( " +
                            "output(count()) all(" +
                                "group(array.at(categories,2)) each( " +
                                    "output(count())" +
                                ")" +
                            ")" +
                        ")" +
                    ")" +
                ")" +
            ")";

        yql += " | " + grouping;

        SimpleQueryBuilder query = new SimpleQueryBuilder("/search/");
        query.add("hits", "0");
        query.add("yql", yql);
        return query.toString();
    }

    public static Map<String, JsonNode> data(SimpleHttpClient client, Map<String, String> properties) {
        return Map.of("categories", client.getJson(query()));
    }

    public static String getCategoryName(String name) {
        return name.substring(name.lastIndexOf('|') + 1);
    }


}
