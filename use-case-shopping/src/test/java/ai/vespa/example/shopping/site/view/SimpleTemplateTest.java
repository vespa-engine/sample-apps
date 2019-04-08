// Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example.shopping.site.view;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

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