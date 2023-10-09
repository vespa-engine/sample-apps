// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.view;

import ai.vespa.example.shopping.site.data.SimpleQueryBuilder;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;
import java.util.Optional;
import java.util.StringJoiner;

import static ai.vespa.example.shopping.site.data.Categories.getCategoryName;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getDoubleValue;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getIntValue;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getStringValue;
import static ai.vespa.example.shopping.site.view.SimpleHtmlBuilder.*;

public class ItemRenderer {

    public static SimpleTemplate render(Map<String, JsonNode> data, Map<String, String> properties) {
        SimpleTemplate template = new SimpleTemplate("item.html.template");
        template.set("asin", properties.getOrDefault("i", ""));
        template.set("page-title", buildItemTitle(data.get("item")));
        template.set("iteminfo", renderItem(data.get("item")));
        template.set("related", renderRelated(data.get("related")));
        template.set("reviews", ReviewRenderer.renderReviews(data.get("reviews"), properties));
        template.set("review-count", getStringValue(data.get("reviews"), "root", "fields", "totalCount").orElse("0") + " reviews");
        template.set("review-distribution", ReviewRenderer.renderReviewDistribution(data.get("reviews")));
        return template;
    }

    private static String buildItemTitle(JsonNode item) {
        return getStringValue(item, "fields", "title").orElse("");
    }

    static String renderItem(JsonNode item) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        Optional<String> image = getStringValue(item, "fields", "images");  // use first image
        Optional<String> title = getStringValue(item, "fields", "title");
        Optional<String> description = getStringValue(item, "fields", "description");
        Optional<Double> price = getDoubleValue(item, "fields", "price");
        Optional<Integer> ratingStars = getIntValue(item, "fields", "rating_stars");
        Optional<Integer> ratingCount = getIntValue(item, "fields", "rating_count");

        StringJoiner sj = new StringJoiner(" | ");
        sj.add("<a href=\"/site\">Home</a>");
        JsonNode categories = item.get("fields").get("categories");
        if (categories != null && categories.size() > 0) {
            for (JsonNode category : categories) {
                String name = getCategoryName(category.asText());
                String href = new SimpleQueryBuilder("/site/search").add("cat", category.asText()).toString();
                String anchor = "<a href=\"" + href + "\">" + name + "</a>";
                sj.add(anchor);
            }
        }

        html.div("item", (level_1) -> {
            html.div("item-categories", (level_2) -> {
                html.text(sj.toString());
            });
            html.div("item-image", (level_2) -> {
                html.img(image.get());
            });
            html.div("item-details", (level_2) -> {
                html.div("item-title", (level_3) -> {
                    html.text(title.get());
                });
                html.div("item-description", (level_3) -> {
                    html.text(description.orElse("No description"));
                });
                html.div("item-price", (level_3) -> {
                    html.text(String.format("$ %.2f", price.get()));
                });
                html.div("item-rating", (level_3) -> {
                    if (ratingCount.get() > 0) {
                        html.text(ratingToStarsEmoji(ratingStars.get(),ratingCount.get()));
                        //html.img(ratingToStarsImage(ratingStars.get(), ratingCount.get()));
                        html.span("item-rating-count", (level_4) -> {
                            html.text(String.format("(%d)", ratingCount.get()));
                        });
                    }
                });
            });
        });
        return html.build();
    }

    static String renderRelated(JsonNode related) {
        if (related == null) return "";

        int totalCount = getIntValue(related, "root", "fields", "totalCount").orElse(0);

        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        html.div("title-container", (level_1) -> {
            html.div("title-text", (level_2) -> {
                html.text(String.format("%d related items", totalCount));
            });
            html.div("title-line", (level_2) -> {
                html.text("");
            });
        });

        if (totalCount > 0) {
            html.div("related-items", (level_1) -> {
                for (JsonNode result : related.get("root").get("children")) {

                    Optional<String> asin = getStringValue(result, "fields", "asin");
                    Optional<String> image = getStringValue(result, "fields", "images");  // use first image
                    Optional<String> title = getStringValue(result, "fields", "title");
                    Optional<Double> price = getDoubleValue(result, "fields", "price");
                    Optional<Integer> ratingStars = getIntValue(result, "fields", "rating_stars");
                    Optional<Integer> ratingCount = getIntValue(result, "fields", "rating_count");

                    String href = new SimpleQueryBuilder("/site/item").add("i", asin.get()).toString();

                    html.div("related-item", (level_2) -> {
                        html.div("related-item-image", (level_3) -> {
                            html.a(href, (level_4) -> {
                                html.img(image.orElse(""));
                            });
                        });
                        html.div("related-item-details", (level_3) -> {
                            html.div("related-item-title", (level_4) -> {
                                html.a(href, (level_5) -> {
                                    html.text(truncate(title.orElse(""), 120));
                                });
                            });
                            html.div("related-item-price", (level_4) -> {
                                html.text(String.format("$ %.2f", price.orElse(0.0)));
                            });
                            html.div("related-item-rating", (level_4) -> {
                                if (ratingCount.get() > 0) {
                                    //html.img(ratingToStarsImage(ratingStars.get(), ratingCount.get()));
                                    html.text(ratingToStarsEmoji(ratingStars.get(),ratingCount.get()));
                                }
                            });

                        });
                    });
                }
            });
        }
        return html.build();
    }

}
