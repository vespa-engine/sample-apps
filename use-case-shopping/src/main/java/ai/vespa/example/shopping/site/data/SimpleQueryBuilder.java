// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.data;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class SimpleQueryBuilder {

    private final String prefix;
    private final Map<String, String> properties;

    public SimpleQueryBuilder() {
        this("");
    }

    public SimpleQueryBuilder(String prefix) {
        this.prefix = prefix;
        this.properties = new HashMap<>();
    }

    public <T> SimpleQueryBuilder add(String key, T value) {
        return add(key, String.valueOf(value));
    }

    public SimpleQueryBuilder add(String key, String value) {
        properties.put(key, URLEncoder.encode(value, StandardCharsets.UTF_8));
        return this;
    }

    public SimpleQueryBuilder add(Map<String, String> properties) {
        properties.forEach(this::add);
        return this;
    }

    public String toString() {
        return prefix + (prefix.length() > 0 && properties.size() > 0 ? "?" : "") + join(properties);
    }

    private static String join(Map<String, String> properties) {
        return properties.entrySet().stream()
                .map(entry -> entry.getKey() + "=" + entry.getValue())
                .collect(Collectors.joining("&"));
    }

}
