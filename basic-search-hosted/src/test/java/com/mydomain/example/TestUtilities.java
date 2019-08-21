package com.mydomain.example;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.TestRuntime;
import com.yahoo.data.access.Inspector;
import com.yahoo.data.access.simple.JsonRender;
import com.yahoo.data.access.slime.SlimeAdapter;
import com.yahoo.slime.JsonDecoder;
import com.yahoo.slime.Slime;
import org.junit.jupiter.api.Assertions;

import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static java.net.URLEncoder.encode;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Collections.emptyMap;

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

    static HttpResponse<String> send(Endpoint endpoint, HttpRequest.Builder request) {
        // TODO jvenstad: Allow endpoint.send(request), inline this, and put request(For) in Endpoint.
        return endpoint.send(request, HttpResponse.BodyHandlers.ofString(UTF_8));
    }

    static HttpRequest.Builder requestFor(Endpoint endpoint, String path) {
        return requestFor(endpoint, path, emptyMap());
    }

    static HttpRequest.Builder requestFor(Endpoint endpoint, String path, Map<String, String> properties) {
        // TODO jvenstad: Remove scheme/port ugliness; it is needed for now, as certificates are not ready.
        URI fixedUri = URI.create("http://" + endpoint.uri().getHost() + ":443/")
                          .resolve(path +
                                   properties.entrySet().stream()
                                             .map(entry -> encode(entry.getKey(), UTF_8) + "=" + encode(entry.getValue(), UTF_8))
                                             .collect(Collectors.joining("&", "?", "")))
                          .normalize();
        return HttpRequest.newBuilder(fixedUri);
    }

}
