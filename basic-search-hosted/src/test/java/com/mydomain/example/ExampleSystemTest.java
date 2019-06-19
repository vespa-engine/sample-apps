package com.mydomain.example;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.TestRuntime;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

@Category(ai.vespa.hosted.cd.FunctionalTest.class) // This will change to SystemTest again.
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
