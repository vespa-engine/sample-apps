// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.example;

import ai.vespa.models.evaluation.ModelsEvaluator;
import ai.vespa.models.evaluation.FunctionEvaluator;
import com.yahoo.container.jdisc.HttpRequest;
import com.yahoo.container.jdisc.HttpResponse;
import com.yahoo.container.jdisc.ThreadedHttpRequestHandler;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.serialization.JsonFormat;

import java.io.IOException;
import java.io.OutputStream;


public class MyHandler extends ThreadedHttpRequestHandler {

    private final ModelsEvaluator modelsEvaluator;

    public MyHandler(ModelsEvaluator modelsEvaluator, Context context) {
        super(context);
        this.modelsEvaluator = modelsEvaluator;
    }

    @Override
    public HttpResponse handle(HttpRequest request) {
        Tensor result = Tensor.from(0.0);

        // Determine which model to evaluate
        String model = request.getProperty("model");
        if (model.equalsIgnoreCase("transformer")) {
            result = evaluateTransformerModel(request);
        }

        // Return result as JSON
        return new RawResponse(JsonFormat.encode(result));
    }

    private Tensor evaluateTransformerModel(HttpRequest request) {
        // Create evaluator
        FunctionEvaluator evaluator = modelsEvaluator.evaluatorOf("transformer");

        // Get the input - this model only has one input
        String inputString = request.getProperty("input");

        // Convert to a Vespa tensor
        Tensor input = Tensor.from(TensorType.fromSpec("tensor<int8>(x[])"), inputString);

        // Here, do any processing of the input - e.g. tokenize or whatever you'd like
        input = Util.renameDimension(input, "x", "d1");
        input = Util.addDimension(input, "d0");

        // Evaluate model
        Tensor result = evaluator.bind("input", input).evaluate();
        return result;
    }

    private static class RawResponse extends HttpResponse {

        private final byte[] data;

        RawResponse(byte[] data) {
            super(200);
            this.data = data;
        }

        @Override
        public String getContentType() {
            return "application/json";
        }

        @Override
        public void render(OutputStream outputStream) throws IOException {
            outputStream.write(data);
        }
    }

}
