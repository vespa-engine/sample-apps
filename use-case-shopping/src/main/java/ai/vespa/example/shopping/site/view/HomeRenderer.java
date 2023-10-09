// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.view;

import ai.vespa.example.shopping.site.data.SimpleQueryBuilder;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;
import java.util.Optional;

import static ai.vespa.example.shopping.site.data.Categories.getCategoryName;
import static ai.vespa.example.shopping.site.view.SimpleHtmlBuilder.ratingToStarsEmoji;
import static ai.vespa.example.shopping.site.view.SimpleHtmlBuilder.truncate;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getDoubleValue;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getIntValue;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getStringValue;

public class HomeRenderer {

    public static SimpleTemplate render(Map<String, JsonNode> data, Map<String, String> properties) {
        SimpleTemplate template = new SimpleTemplate("home.html.template");
        template.set("promoted-items", renderPromotedItems(data.get("promoted")));
        template.set("home-navigation", renderCategories(data.get("categories")));
        template.set("page-title", "Home");
        return template;
    }

    static String renderPromotedItems(JsonNode root) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        html.div("promoted-items", (level_1) -> {
            for (JsonNode result : root.get("root").get("children")) {

                Optional<String>  asin = getStringValue(result, "fields", "asin");
                Optional<String>  image = getStringValue(result, "fields", "images");  // use first image
                Optional<String>  title = getStringValue(result, "fields", "title");
                Optional<Double>  price = getDoubleValue(result, "fields", "price");
                Optional<Integer> ratingStars = getIntValue(result, "fields", "rating_stars");
                Optional<Integer> ratingCount = getIntValue(result, "fields", "rating_count");

                String href = "/site/item?i=" + asin.get();

                html.div("promoted-item", (level_2) -> {
                    html.div("promoted-item-image", (level_3) -> {
                        html.a(href, (level_4) -> {
                            html.img(image.get());
                        });
                    });
                    html.div("promoted-item-details", (level_3) -> {
                        html.div("promoted-item-title", (level_4) -> {
                            html.a(href, (level_5) -> {
                                html.text(truncate(title.orElse(""), 120));
                            });
                        });
                        html.div("promoted-item-price", (level_4) -> {
                            html.text(String.format("$ %.2f", price.get()));
                        });
                        html.div("promoted-item-rating", (level_4) -> {
                            if (ratingCount.get() > 0) {
                                html.text(String.format("%s", ratingToStarsEmoji(ratingStars.get(), ratingCount.get())));
                                //html.img(SimpleHtmlBuilder.ratingToStarsImage(ratingStars.get(), ratingCount.get()));
                                html.div("promoted-item-rating-count", (level_5) -> {
                                    html.text(String.format("(%d)", ratingCount.get()));
                                });
                            }
                        });
                    });
                });
            }
        });
        return html.build();
    }

    static String renderCategories(JsonNode root) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        for (JsonNode cat0 : root.get("root").get("children").get(0).get("children").get(0).get("children")) {
            if (cat0.get("fields").get("count()").asInt() <= 1) {
                continue;
            }

            html.div("home-navigation-header-1", (level_1) -> {
                String name = getCategoryName(cat0.get("value").asText());
                html.text(name);
            });

            for (JsonNode cat1 : cat0.get("children").get(0).get("children")) {
                html.div("home-navigation-header-2", (level_2) -> {
                    String name = getCategoryName(cat1.get("value").asText());
                    html.text(name);
                });

                for (JsonNode cat2 : cat1.get("children").get(0).get("children")) {
                    String name = cat2.get("value").asText();
                    String href = new SimpleQueryBuilder("/site/search").add("cat", name).toString();
                    html.div("home-navigation-header-3", (level_3) -> {
                        html.a(href, (level_4) -> {
                            html.text(getCategoryName(name));
                        });
                    });
                }
            }
        }

        return html.build();
    }

}
