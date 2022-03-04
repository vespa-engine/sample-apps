// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.text_search.site;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

import ai.vespa.example.text_search.site.view.SimpleTemplate;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class SimpleTemplateTest {

    @Test
    public void testTemplateRendering() {
        SimpleTemplate template = new SimpleTemplate("test.html.template");
        template.set("nested", "{{nested}}Title:{{title}}");
        template.set("title", "Title");
        template.set("content", "Content");

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        template.render(stream);

        assertEquals("Title:Title\nContent:Content", stream.toString(StandardCharsets.UTF_8));
    }

}