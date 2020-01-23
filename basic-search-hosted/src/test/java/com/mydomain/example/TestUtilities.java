package com.mydomain.example;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.TestRuntime;
import com.yahoo.data.access.Inspector;
import com.yahoo.data.access.simple.JsonRender;
import com.yahoo.data.access.slime.SlimeAdapter;
import com.yahoo.slime.JsonDecoder;
import com.yahoo.slime.Slime;
import org.junit.jupiter.api.Assertions;

import java.net.http.HttpResponse;
import java.util.Map.Entry;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Base class for functional tests of a Vespa deployment, containing only common code.
 *
 * @author jonmv
 */
class TestUtilities {

    /** The endpoint of the "container" cluster, as per services.xml. */
    static Endpoint container() {
        return TestRuntime.get().deploymentToTest().endpoint("container");
    }

    static void assertStatusCode(int expected, HttpResponse<String> response) {
        assertStatusCode(expected, response, "");
    }

    static void assertStatusCode(int expected, HttpResponse<String> response, String reasoning) {
        Assertions.assertEquals(expected,
                                response.statusCode(),
                                (reasoning.isBlank() ? "" : reasoning + ":\n") + response + "\n" + response.body());
    }

    static Inspector toInspector(String json) {
        return new SlimeAdapter(new JsonDecoder().decode(new Slime(), json.getBytes(UTF_8)).get());
    }

    static void print(Inspector inspector) {
        System.err.println(JsonRender.render(inspector, new StringBuilder(), false));
    }

    static Stream <Inspector> entriesOf(Inspector inspector) {
        return StreamSupport.stream(Spliterators.spliterator(inspector.entries().iterator(),
                                                             inspector.entryCount(),
                                                             Spliterator.ORDERED),
                                    false);
    }

    static Stream <Entry<String, Inspector>> fieldsOf(Inspector inspector) {
        return StreamSupport.stream(Spliterators.spliterator(inspector.fields().iterator(),
                                                             inspector.fieldCount(),
                                                             Spliterator.ORDERED),
                                    false);
    }

}
