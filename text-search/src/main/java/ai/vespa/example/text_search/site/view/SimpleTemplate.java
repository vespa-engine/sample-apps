// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.text_search.site.view;

import com.yahoo.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SimpleTemplate {

    private final static Pattern pattern = Pattern.compile("\\{\\{([a-z0-9\\- ]*)\\}\\}", Pattern.MULTILINE | Pattern.CASE_INSENSITIVE);

    private final String template;
    private final Map<String, String> values = new HashMap<>();

    public SimpleTemplate(String templateName) {
        this.template = getTemplate(templateName);
        addDefaultValues();
    }

    public void set(String tag, String value) {
        if (value != null) {
            values.put(tag, value);
        }
    }

    public void render(OutputStream stream) {
        try (PrintWriter writer = new PrintWriter(stream)) {
            render(writer, template, new ArrayDeque<>());
        }
    }

    private void render(PrintWriter writer, String template, Deque<String> stack) {
        Matcher m = pattern.matcher(template);
        int start = 0;
        while (m.find(start)) {
            writer.print(template.substring(start, m.start()));
            String tag = m.group(1);
            if (values.containsKey(tag) && ! stack.contains(tag)) {  // try to avoid infinite recursion
                stack.push(tag);
                render(writer, values.get(tag), stack);
                stack.pop();
            }
            start = m.end();
        }
        writer.print(template.substring(start));
    }

    private String getTemplate(String name) {
        String filename = "templates/" + name;
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(filename)) {
            return IOUtils.readAll(is, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void addDefaultValues() {
        set("header", getTemplate("header.html.template"));
        set("footer", getTemplate("footer.html.template"));
    }

}
