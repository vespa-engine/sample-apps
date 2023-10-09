// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

package ai.vespa.example;

import ai.vespa.models.evaluation.ModelsEvaluator;
import com.yahoo.container.jdisc.HttpRequest;
import com.yahoo.container.jdisc.HttpResponse;
import com.yahoo.container.jdisc.ThreadedHttpRequestHandler;
import com.yahoo.jdisc.http.HttpRequest.Method;
import com.yahoo.vespa.model.container.ml.ModelsEvaluatorTester;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class MyHandlerTest {

    @Test
    public void testMyHandler() throws IOException {
        ModelsEvaluator modelsEvaluator = ModelsEvaluatorTester.create("src/main/application/models");
        MyHandler myHandler = new MyHandler(modelsEvaluator, ThreadedHttpRequestHandler.testContext());

        Map<String, String> properties = new HashMap<>();
        properties.put("model", "transformer");
        properties.put("input", "{{x:0}:1,{x:1}:2,{x:2}:3}");
        HttpRequest request = HttpRequest.createTestRequest("", Method.GET, InputStream.nullInputStream(), properties);
        HttpResponse response = myHandler.handle(request);

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        response.render(stream);
        assertTrue(stream.toString().contains("\"address\":{\"d0\":\"0\",\"d1\":\"0\",\"d2\":\"0\"},\"value\":1.64956"));
    }

}
