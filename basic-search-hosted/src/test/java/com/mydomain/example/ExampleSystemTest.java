package com.mydomain.example;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.SystemTest;
import ai.vespa.hosted.cd.TestRuntime;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

@Tag(SystemTest.name)
public class ExampleSystemTest {

    @Test
    public void checkStatusHtml() throws IOException {
        Endpoint endpoint = TestRuntime.get().deploymentToTest().endpoint("container");
        URI fixedUrl = URI.create("http://" + endpoint.uri().getHost() + ":443/"); // Certificates not yet ready.
        endpoint.send(HttpRequest.newBuilder(fixedUrl.resolve("/status.html")),
                      HttpResponse.BodyHandlers.ofInputStream())
                .body().transferTo(System.out);
    }

}
