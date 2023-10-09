// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.data;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;
import java.util.StringJoiner;

import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getNode;

public class Item {

    static String queryItem(Map<String, String> properties) {
        SimpleQueryBuilder query = new SimpleQueryBuilder("/search/");
        query.add("yql", "select * from sources item where asin contains \"" + properties.get("i") + "\"");
        query.add("summary", "default");
        query.add("hits", 1);
        return query.toString();
    }

    static String queryRelated(JsonNode item) {
        JsonNode asins = item.get("fields").get("related");
        if (asins == null) {
            return null;
        }
        StringJoiner sj = new StringJoiner(" or ");
        for (JsonNode asin : asins) {
            sj.add("asin contains \"" + asin.asText() + "\"");
        }
        SimpleQueryBuilder query = new SimpleQueryBuilder("/search/");
        query.add("yql", "select * from sources item where " + sj);
        query.add("summary", "short");
        query.add("hits", 20);
        return query.toString();
    }

    public static Map<String, JsonNode> data(SimpleHttpClient client, Map<String, String> properties) {
        JsonNode item = client.getJson(queryItem(properties));
        item = getNode(item, "root", "children").get();
        JsonNode related = client.getJson(queryRelated(item));
        return Map.of("item", item, "related", related);
    }

}
