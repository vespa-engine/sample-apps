// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.view;

import ai.vespa.example.shopping.site.data.SimpleQueryBuilder;
import com.fasterxml.jackson.databind.JsonNode;

import java.util.Map;
import java.util.Optional;

import static ai.vespa.example.shopping.site.view.SimpleHtmlBuilder.ratingToStarsEmoji;
import static ai.vespa.example.shopping.site.view.SimpleHtmlBuilder.ratingToStarsImage;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getIntValue;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getNode;
import static ai.vespa.example.shopping.site.data.SimpleHttpClient.getStringValue;

public class ReviewRenderer {

    static String renderReviews(JsonNode root, Map<String, String> properties) {
        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        if ( ! getNode(root, "root", "children").isPresent()) {
            return "";
        }

        html.div("reviews", (level_1) -> {
            for (JsonNode review : root.get("root").get("children")) {

                // Skip grouping result here
                if (getStringValue(review, "id").get().startsWith("group")) {
                    continue;
                }

                Optional<String>  id = getStringValue(review, "fields", "reviewer_id");
                Optional<String>  name = getStringValue(review, "fields", "reviewer_name");
                Optional<String>  title = getStringValue(review, "fields", "title");
                Optional<String>  text = getStringValue(review, "fields", "text");
                Optional<Integer> stars = getIntValue(review, "fields", "stars");
                Optional<Integer> upvotes = getIntValue(review, "fields", "upvotes");
                Optional<Integer> downvotes = getIntValue(review, "fields", "downvotes");

                int totalVotes = upvotes.orElse(0) + downvotes.orElse(0);

                SimpleQueryBuilder query = new SimpleQueryBuilder("/site/vote").add(properties).add("rid", id.get());
                String upvotehref = query.add("dir", "1").toString();
                String downvotehref = query.add("dir", "-1").toString();

                html.div("review", (level_2) -> {
                    html.div("review-title", (level_3) -> {
                        html.text(title.orElse(""));
                    });
                    html.div("review-name", (level_3) -> {
                        html.text("By " + name.orElse("unknown"));
                    });
                    html.div("review-text", (level_3) -> {
                        html.text(text.orElse(""));
                    });
                    html.div("review-stars", (level_3) -> {
                        html.text(ratingToStarsEmoji(stars.get(),1));
                        //html.img(ratingToStarsImage(stars.get(), 1));
                    });
                    html.div("review-helpful", (level_3) -> {
                        html.text(upvotes.get() + " of " + totalVotes + " people found this helpful. Did you?");
                        html.a(upvotehref, (level_4) -> {
                            html.text("yes");
                        });
                        html.text(" | ");
                        html.a(downvotehref, (level_4) -> {
                            html.text("no");
                        });
                    });
                });
            }
        });

        return html.build();
    }

    public static String renderReviewDistribution(JsonNode root) {
        if (root == null) return "";
        if ( ! getNode(root, "root", "children", "children").isPresent()) return "";

        Optional<Integer> totalCount = getIntValue(root, "root", "fields", "totalCount");
        if (totalCount.orElse(0) == 0) {
            return "";
        }

        SimpleHtmlBuilder html = new SimpleHtmlBuilder();

        html.div("review-distribution", (level_1) -> {
            JsonNode grouping = null;
            for (JsonNode child : root.get("root").get("children")) {
                if (getStringValue(child, "id").get().startsWith("group")) {
                    grouping = child;
                    break;
                }
            }
            if (grouping == null) {
                throw new RuntimeException("Could not find grouping node");
            }

            int[] counts = new int[5];
            for (JsonNode review : grouping.get("children").get(0).get("children")) {
                Optional<Integer> value = getIntValue(review, "value");
                Optional<Integer> count = getIntValue(review, "fields", "count()");
                counts[value.get() - 1] = count.get();
            }

            for (int i = 5; --i >= 0; ) {
                int star = i + 1;
                int count = counts[i];
                String percent = String.format("%.0f%%", count * 100.0f / totalCount.get());

                html.div("review-distribution-star", (level_2) -> {
                    html.div("review-distribution-star-desc", (level_3) -> {
                        html.text(star + " star");
                    });
                    html.div("review-distribution-star-percent", (level_3) -> {
                        html.text(percent);
                    });
                    html.element("div", Map.of("class", "review-distribution-star-bar", "style", "width: " + percent + ";"));
                    html.element("div", Map.of("class", "review-distribution-star-bar-back", "style", "width: 100%;"));
                });

            }

        });
        return html.build();
    }


}
