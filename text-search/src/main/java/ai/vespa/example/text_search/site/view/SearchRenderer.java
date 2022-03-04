// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.text_search.site.view;

import ai.vespa.example.text_search.site.data.SimpleQueryBuilder;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;
import java.util.Optional;
import java.util.StringJoiner;

import static ai.vespa.example.text_search.site.data.SimpleHttpClient.getIntValue;
import static ai.vespa.example.text_search.site.data.SimpleHttpClient.getStringValue;
import static ai.vespa.example.text_search.site.view.SimpleHtmlBuilder.truncate;

public class SearchRenderer {

    public static SimpleTemplate render(Map<String, JsonNode> data, Map<String, String> properties) {
        JsonNode results = data.get("searchresults");
        SimpleTemplate template = new SimpleTemplate("search.html.template");
        template.set("page-title", buildPageTitle(properties));
        template.set("search-query", properties.getOrDefault("q", ""));
        template.set("search-query-parameters", new SimpleQueryBuilder().add(properties).toString());
        template.set("search-profiles", renderProfileOptions(properties));
        template.set("search-results-count", String.valueOf(totalCount(results)));
        template.set("search-results-pagination", renderPagination(properties, results));
        template.set("results", renderResults(results));
        return template;
    }

    static String buildPageTitle(Map<String, String> properties) {
        StringJoiner sj = new StringJoiner(" and ");
        if (properties.containsKey("q") && properties.get("q").length() > 0) {
            sj.add("'" + properties.get("q") + "'");
        }
        return "Results for " + sj.toString() + "";
    }

    static String renderResults(JsonNode root) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        if (totalCount(root) > 0 && root.get("root").has("children")) {
            html.div("search-items", (level_1) -> {
                for (JsonNode result : root.get("root").get("children")) {

                    Optional<String> title = getStringValue(result, "fields", "title");
                    Optional<String> body  = getStringValue(result, "fields", "body");
                    Optional<String> url   = getStringValue(result, "fields", "url");

                    if (title.isPresent() && title.get().strip().length() > 0 && body.isPresent() && url.isPresent()) {
                        html.div("search-item", (level_2) -> {
                            html.div("search-item-title", (level_3) -> {
                                html.a(url.get(), (level_4) -> {
                                    html.text(truncate(title.get(), 250));
                                });
                            });
                            html.div("search-item-body", (level_3) -> {
                                html.text(body.get());
                            });
                            html.div("search-item-url", (level_3) -> {
                                html.a(url.get(), (level_4) -> {
                                    html.text(url.get());
                                });
                            });
                        });
                    }
                }
            });
        }
        return html.build();
    }

    static String renderPagination(Map<String, String> properties, JsonNode root) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        int page = Integer.parseInt(properties.getOrDefault("p", "0"));
        int resultsPerPage = Integer.parseInt(properties.getOrDefault("h", "10"));

        SimpleQueryBuilder query = new SimpleQueryBuilder("/site/search/").add(properties);
        final String previousPageUrl = query.add("p", page - 1).toString();
        final String nextPageUrl = query.add("p", page + 1).toString();

        if (page > 0) {
            html.div("search-results-previous", (level_1) -> {
                html.a(previousPageUrl, (level_2) -> {
                    html.text("Previous page");
                });
            });
        }
        if (page < Math.floor((double) totalCount(root) / resultsPerPage)) {
            html.div("search-results-next", (level_1) -> {
                html.a(nextPageUrl, (level_2) -> {
                    html.text("Next page");
                });
            });
        }
        return html.build();
    }

    static int totalCount(JsonNode root) {
        Optional<Integer> totalCount = getIntValue(root, "root", "fields", "totalCount");
        if (!totalCount.isPresent()) {
            throw new RuntimeException("Unable to retrieve hit count");
        }
        return totalCount.get();
    }

    static String renderProfileOptions(Map<String, String> properties) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        String selectedProfile = properties.getOrDefault("profile", "bm25");
        html.option("default", selectedProfile.equalsIgnoreCase("default"));
        html.option("bm25", selectedProfile.equalsIgnoreCase("bm25"));

        return html.build();
    }

}
