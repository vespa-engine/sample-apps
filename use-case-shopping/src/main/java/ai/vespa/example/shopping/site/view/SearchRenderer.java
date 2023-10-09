// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.view;

import ai.vespa.example.shopping.site.data.SimpleQueryBuilder;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.StringJoiner;

import static ai.vespa.example.shopping.site.data.Categories.getCategoryName;
import static ai.vespa.example.shopping.site.view.SimpleHtmlBuilder.ratingToStarsImage;
import static ai.vespa.example.shopping.site.view.SimpleHtmlBuilder.ratingToStarsEmoji;
import static ai.vespa.example.shopping.site.view.SimpleHtmlBuilder.truncate;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getDoubleValue;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getIntValue;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getStringValue;

public class SearchRenderer {

    public static SimpleTemplate render(Map<String, JsonNode> data, Map<String, String> properties) {
        JsonNode results = data.get("searchresults");
        SimpleTemplate template = new SimpleTemplate("search.html.template");
        template.set("page-title", buildPageTitle(properties));
        String query = properties.getOrDefault("q", "");
        if(!query.isBlank())
            query = query.replace("\"","&quot;");
        template.set("search-query", query);
        template.set("search-query-parameters", new SimpleQueryBuilder().add(properties).toString());
        template.set("search-results-count", String.valueOf(totalCount(results)));
        template.set("search-results-pagination", renderPagination(properties, results));
        template.set("results-grouping", renderSearchNavigation(properties, results));
        template.set("results", renderResults(results));
        return template;
    }

    static String buildPageTitle(Map<String, String> properties) {
        StringJoiner sj = new StringJoiner(" and ");
        if (properties.containsKey("q") && properties.get("q").length() > 0) {
            sj.add("'" + properties.get("q") + "'");
        }
        if (properties.containsKey("cat") && properties.get("cat").length() > 0) {
            sj.add("category '" + getCategoryName(properties.get("cat")) + "'");
        }
        if (properties.containsKey("brand") && properties.get("brand").length() > 0) {
            sj.add("brand '" + properties.get("brand") + "'");
        }
        if (properties.containsKey("stars") && properties.get("stars").length() > 0) {
            sj.add(properties.get("stars") + " star rating");
        }
        if (properties.containsKey("price_from") && properties.get("price_from").length() > 0) {
            sj.add("price &gt; " + properties.get("price_from"));
        }
        if (properties.containsKey("price_to") && properties.get("price_to").length() > 0 && properties.get("price_to").length() < 9) {
            sj.add("price &lt; " + properties.get("price_to"));
        }
        return "Results for " + sj.toString() + "";
    }

    static String renderResults(JsonNode root) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        if (totalCount(root) > 0) {
            html.div("search-items", (level_1) -> {
                for (JsonNode result : root.get("root").get("children")) {

                    // Skip grouping result here - it is rendered in search navigation
                    if (getStringValue(result, "id").get().startsWith("group")) {
                        continue;
                    }

                    Optional<String> asin = getStringValue(result, "fields", "asin");
                    Optional<String> brand = getStringValue(result, "fields", "brand");
                    Optional<String> image = getStringValue(result, "fields", "images");  // use first image
                    Optional<String> title = getStringValue(result, "fields", "title");
                    Optional<Double> price = getDoubleValue(result, "fields", "price");
                    Optional<Integer> ratingStars = getIntValue(result, "fields", "rating_stars");
                    Optional<Integer> ratingCount = getIntValue(result, "fields", "rating_count");

                    Optional<Double> semanticScore = getDoubleValue(result, "fields", "matchfeatures", "semantic");
                    Optional<Double> keywordScore = getDoubleValue(result, "fields", "matchfeatures", "keyword");
                    Optional<Double> ratingScore = getDoubleValue(result, "fields", "matchfeatures", "rating");

                    String href = new SimpleQueryBuilder("/site/item").add("i", asin.get()).toString();

                    html.div("search-item", (level_2) -> {
                        html.div("search-item-image", (level_3) -> {
                            html.a(href, (level_4) -> {
                                html.img(image.get());
                            });
                        });
                        html.div("search-item-title", (level_3) -> {
                            html.a(href, (level_4) -> {
                                if(brand.isPresent() && title.isPresent() && !title.get().startsWith(brand.get())) {
                                    html.text(String.format("%s - %s", brand.orElse(""), truncate(title.orElse(""), 128)));
                                } else {
                                    html.text(String.format("%s", truncate(title.orElse(""), 128)));
                                }
                            });
                        });
                        html.div("search-item-scores", (level_3) -> {
                            html.text(String.format("Semantic score %.2f, Keyword score %.2f, Rating score %.2f",
                                    semanticScore.get(), keywordScore.get(), ratingScore.get()));
                        });
                        html.div("search-item-price", (level_3) -> {
                            html.text(String.format("$ %.2f", price.get()));
                        });
                        html.div("search-item-rating", (level_3) -> {
                            if (ratingCount.get() > 0) {
                                html.text(String.format("%s",ratingToStarsEmoji(ratingStars.get())));
                                //html.img(ratingToStarsImage(ratingStars.get(), ratingCount.get()));
                                html.span("search-rating-count", (level_4) -> {
                                      html.text(String.format("(%d)", ratingCount.get()));
                                });
                            }
                        });
                    });
                }
            });
        }
        return html.build();
    }

    static String renderPagination(Map<String, String> properties, JsonNode root) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        int page = Integer.parseInt(properties.getOrDefault("p", "0"));
        int resultsPerPage = Integer.parseInt(properties.getOrDefault("r", "10"));

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

    static String renderSearchNavigation(Map<String, String> properties, JsonNode root) {
        final Map<String, JsonNode> map = new HashMap<>();
        if (totalCount(root) > 0) {
            for (JsonNode result : root.get("root").get("children")) {

                // Skip normal hits here
                if (!getStringValue(result, "id").get().startsWith("group")) {
                    continue;
                }
                if (result.get("children") == null) {
                    continue;
                }

                for (JsonNode group : result.get("children")) {
                    if (group.get("label").asText().equalsIgnoreCase("categories")) {
                        map.put("categories", group);
                    }
                    if (group.get("label").asText().equalsIgnoreCase("brand")) {
                        map.put("brands", group);
                    }
                    if (group.get("label").asText().equalsIgnoreCase("div(rating_stars, rating_count)")) {
                        map.put("stars", group);
                    }
                    if (group.get("label").asText().startsWith("predefined(price")) {
                        map.put("price", group);
                    }
                }
            }
        }

        SimpleHtmlBuilder html = new SimpleHtmlBuilder();
        renderCategories(html, properties, map.get("categories"));
        renderBrands(html, properties, map.get("brands"));
        renderStars(html, properties, map.get("stars"));
        renderPrice(html, properties, map.get("price"));
        return html.build();
    }

    private static void renderCategories(SimpleHtmlBuilder html, Map<String, String> properties, JsonNode categories) {
        if (categories == null) {
            return;
        }
        html.div("search-groups-header", (level_1) -> {
            html.text("Categories");
        });
        html.div("search-groups-list", (level_1) -> {
            int max = 20;
            for (JsonNode result : categories.get("children")) {
                if (--max < 0) {
                    break;
                }
                Optional<Integer> count = getIntValue(result, "fields", "count()");
                Optional<String> value = getStringValue(result, "value");
                if (!value.isPresent()) {
                    continue;
                }
                String name = getCategoryName(value.get());
                String href = new SimpleQueryBuilder("/site/search").add(properties).add("cat", value.get()).toString();

                html.div("search-groups-item", (level_2) -> {
                    html.a(href, (level_3) -> {
                        html.text(String.format("%s", name));
                    });
                });
            }
        });
    }

    private static void renderBrands(SimpleHtmlBuilder html, Map<String, String> properties, JsonNode brands) {
        if (brands == null) {
            return;
        }
        html.div("search-groups-header", (level_1) -> {
            html.text("Brands");
        });
        html.div("search-groups-list", (level_1) -> {
            int max = 20;
            for (JsonNode result : brands.get("children")) {
                if (--max < 0) {
                    break;
                }
                Optional<Integer> count = getIntValue(result, "fields", "count()");
                Optional<String> value = getStringValue(result, "value");
                if (count.orElse(0) == 0 || value.orElse("").length() == 0) {
                    continue;
                }
                String href = new SimpleQueryBuilder("/site/search").add(properties).add("brand", value.get()).toString();

                html.div("search-groups-item", (level_2) -> {
                    html.a(href, (level_3) -> {
                        html.text(String.format("%s", value.get()));
                    });
                });
            }
        });
    }

    private static void renderStars(SimpleHtmlBuilder html, Map<String, String> properties, JsonNode root) {
        if (root == null) {
            return;
        }
        html.div("search-groups-header", (level_1) -> {
            html.text("Rating");
        });
        html.div("search-groups-list", (level_1) -> {

            int[] stars = new int[6];
            for (JsonNode review : root.get("children")) {
                Optional<Integer> value = getIntValue(review, "value");
                Optional<Integer> count = getIntValue(review, "fields", "count()");
                stars[value.get()] = count.get();
            }

            for (int i = 6; --i > 0; ) {
                int count = stars[i];
                if (count == 0) {
                    continue;
                }
                final int rating = i;
                final String href = new SimpleQueryBuilder("/site/search").add(properties).add("stars", rating).toString();

                html.div("search-groups-item", (level_2) -> {
                    html.a(href, (level_3) -> {
                        html.text(String.format("%s %s", ratingToStarsEmoji(rating), rating));
                    });
                });
            }
        });
    }

    private static void renderPrice(SimpleHtmlBuilder html, Map<String, String> properties, JsonNode price) {
        if (price == null) {
            return;
        }
        html.div("search-groups-header", (level_1) -> {
            html.text("Price range");
        });
        html.div("search-groups-list", (level_1) -> {
            for (JsonNode priceRange : price.get("children")) {

                Optional<String> from = getStringValue(priceRange, "limits", "from");
                Optional<String> to = getStringValue(priceRange, "limits", "to");
                Optional<Integer> count = getIntValue(priceRange, "fields", "count()");
                if (!from.isPresent() || !to.isPresent() || !count.isPresent() || count.get() == 0) {
                    continue;
                }
                //double min = getDoubleValue(priceRange, "fields", "min(price)").get();
                //double max = getDoubleValue(priceRange, "fields", "max(price)").get();
                double min = getDoubleValue(priceRange, "limits", "from").get();
                double max = getDoubleValue(priceRange, "limits", "to").get();
                if (min < 0 || max > 100000)
                    continue;
                String href = new SimpleQueryBuilder("/site/search")
                        .add(properties)
                        .add("price_from", from.get())
                        .add("price_to", to.get())
                        .toString();

                html.div("search-groups-item", (level_2) -> {
                    html.a(href, (level_3) -> {
                        html.text(String.format("$%.0f to $%.0f", min, max));
                    });
                });
            }
        });
    }

}
