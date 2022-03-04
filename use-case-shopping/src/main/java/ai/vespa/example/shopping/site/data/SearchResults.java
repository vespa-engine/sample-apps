// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.data;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;
import java.util.StringJoiner;

public class SearchResults {

    static String query(Map<String, String> properties) {
        SimpleQueryBuilder query = new SimpleQueryBuilder("/search/");

        String yql = "select * from sources item where ";

        StringJoiner where = new StringJoiner(" and ");
        if (properties.containsKey("q")) {
            String q = properties.get("q");
            String userInput = "userInput(\"" + q + "\")";
            String brand = "brand contains \"" + q + "\"";
            where.add(String.format("( %s or %s )", userInput, brand));
        }
        if (properties.containsKey("cat")) {
            where.add("categories contains \"" + properties.get("cat") + "\"");
        }
        if (properties.containsKey("brand")) {
            where.add("brand contains \"" + properties.get("brand") + "\"");
        }
        if (properties.containsKey("price_from")) {
            where.add("price >= " + properties.get("price_from"));
        }
        if (properties.containsKey("price_to")) {
            if (properties.get("price_to").length() < 9) {
                where.add("price <= " + properties.get("price_to"));  // don't set for infinite high price
            }
        }
        yql += where.toString();

        String groupingBrand = "all(group(brand) order(-count()) each(output(count())))";
        String groupingStars = "all(group(rating_stars / rating_count) each(output(count())))";
        String groupingCategories = "all(group(categories) order(-count()) each(output(count())))";
        String groupingPrice = "all( group(predefined(price,bucket[0,10>,bucket[10,25>,bucket[25,50>,bucket[50,100>,bucket[100,200>,bucket[200,500>,bucket[500,inf>)) order(min(price)) each(output(max(price),min(price),count())))";
        String grouping = String.format("all( %s %s %s %s )", groupingBrand, groupingCategories, groupingStars, groupingPrice);
        yql += " | " + grouping;

        query.add("yql", yql);
        query.add("summary", "short");
        query.add("hits", properties.getOrDefault("r", "10"));

        // Price is an attribute, so we should use "order by" to sort. However,
        // since the rating filter depends upon a rank expression, we need to
        // implement sorting in ranking to combine sorting and filtering.
        switch (properties.getOrDefault("sort", "")) {
            case "pl":
                query.add("ranking", "sort_by_price");
                query.add("ranking.features.query(sort_direction)", "-1");
                break;
            case "ph":
                query.add("ranking", "sort_by_price");
                query.add("ranking.features.query(sort_direction)", "1");
                break;
            case "ll":
                query.add("ranking", "sort_by_rating");
                query.add("ranking.features.query(sort_direction)", "-1");
                break;
            case "rh":
                query.add("ranking", "sort_by_rating");
                query.add("ranking.features.query(sort_direction)", "1");
                break;
            default:
                query.add("ranking", "item");
        }

        if (properties.containsKey("p")) {
            int pageVal = Integer.parseInt(properties.get("p"));
            int perPage = Integer.parseInt(properties.getOrDefault("r", "10"));
            query.add("offset", pageVal * perPage);
        }

        if (properties.containsKey("stars")) {
            query.add("ranking.features.query(use_rating_filter)", "1");
            query.add("ranking.features.query(rating)", properties.get("stars"));
        }

        return query.toString();
    }

    public static Map<String, JsonNode> data(SimpleHttpClient client, Map<String, String> properties) {
        return Map.of("searchresults", client.getJson(query(properties)));
    }

}
