// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.text_search.site.view;

import java.util.Map;
import java.util.function.Consumer;

public class SimpleHtmlBuilder {

    private final StringBuilder sb;

    public SimpleHtmlBuilder() {
        sb = new StringBuilder();
    }

    public void element(String elem, Map<String, String> attrs) {
        element(elem, attrs, null);
    }

    public void element(String elem, Map<String, String> attrs, Consumer<SimpleHtmlBuilder> inner) {
        sb.append("<").append(elem).append(" ");
        if (attrs != null) {
            attrs.forEach((k,v) -> sb.append(k).append("=\"").append(v).append("\" "));
        }
        sb.append(">\n");
        if (inner != null) {
            inner.accept(this);
        }
        sb.append("</").append(elem).append(">\n");
    }

    public void div(String cls, Consumer<SimpleHtmlBuilder> inner) {
        element("div", Map.of("class", cls), inner);
    }

    public void a(String href, Consumer<SimpleHtmlBuilder> inner) {
        element("a", Map.of("href", href), inner);
    }

    public void img(String src) {
        element("img", Map.of("src", src), null);
    }

    public void span(String cls, Consumer<SimpleHtmlBuilder> inner) {
        element("span", Map.of("class", cls), inner);
    }

    public void text(String text) {
        sb.append(text);
    }

    public void option(String value, boolean selected) {
        if (selected) {
            element("option", Map.of("value", value, "selected", "true"), (v) -> text(value));
        } else {
            element("option", Map.of("value", value), (v) -> text(value));
        }
    }

    public String build() {
        return sb.toString();
    }

    public static String truncate(String str, int max) {
        if (str.length() < max)
            return str;
        int lastSpace = str.lastIndexOf(" ", max - 4);
        return str.substring(0, lastSpace < 0 ? max - 4 : lastSpace) + " ...";
    }

}
