// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.cloud.playground;

import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.searchlib.rankingexpression.RankingExpression;
import com.yahoo.searchlib.rankingexpression.parser.ParseException;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorAddress;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ProtonExpressionEvaluator {

    public static String evaluate(String json) {
        try {
            ObjectMapper m = new ObjectMapper();
            JsonNode root = m.readTree(json);
            if (root == null) {
                return error("Could not parse json to evaluate");
            }
            if (!root.isArray()) {
                return error("Input is not an array");
            }

            ArrayList<JsonNode> cells = new ArrayList<>();
            for (int i = 0; i < root.size(); ++i) {
                cells.add(root.get(i));
            }

            String jsonInput = createJsonInput(cells);
            String jsonOutput = callVespaEvalExpr(jsonInput);
            String result = prepareOutput(jsonOutput, cells);

            return result;

        } catch (IOException | InterruptedException e) {
            return error(e.getMessage());
        }
    }

    private static String createJsonInput(List<JsonNode> cells) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        JsonGenerator g = new JsonFactory().createGenerator(out, JsonEncoding.UTF8);
        g.writeStartArray();
        for (JsonNode cell : cells) {
            g.writeStartObject();
            g.writeStringField("name", cell.get("name").asText());
            g.writeBooleanField("verbose", cell.get("verbose").asBoolean());
            try {
                g.writeStringField("expr", toPrimitive(cell.get("expr").asText()));
            } catch (ParseException e) {
                g.writeStringField("expr", cell.get("expr").asText());  // error reporting is handled below
            }
            g.writeEndObject();
        }
        g.writeEndArray();
        g.close();
        return out.toString();
    }

    private static String callVespaEvalExpr(String jsonInput) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder("vespa-eval-expr", "json-repl");
        processBuilder.redirectErrorStream(true);
        StringBuilder output = new StringBuilder();
        Process process = processBuilder.start();

        // Write json array to process' stdin
        OutputStream os = process.getOutputStream();
        os.write(jsonInput.getBytes(StandardCharsets.UTF_8));
        os.close();

        // Read output from stdout/stderr
        InputStream inputStream = process.getInputStream();
        while (true) {
            int b = inputStream.read();
            if (b == -1) break;
            output.append((char)b);
        }
        int returnCode = process.waitFor();
        if (returnCode != 0 && output.length() == 0) {
            throw new IOException("No output from 'vespa-eval-expr'. Return code: " + returnCode);
        }
        return output.toString();
    }

    private static String prepareOutput(String json, List<JsonNode> cells) throws IOException {
        ObjectMapper m = new ObjectMapper();
        JsonNode root = m.readTree(json);
        if (! root.isArray() || root.size() != cells.size()) {
            return error("Unexpected return from 'vespa-eval-expr'");
        }

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        JsonGenerator g = new JsonFactory().createGenerator(out, JsonEncoding.UTF8);
        g.writeStartArray();

        for (int i = 0; i < root.size(); ++i) {
            int cellId = cells.get(i).get("cell").asInt();

            g.writeStartObject();
            g.writeNumberField("cell", cellId);

            String expression = cells.get(i).get("expr").asText();
            try {
                g.writeStringField("primitive", toPrimitive(expression));
                if (root.get(i).has("error")) {
                    g.writeStringField("error", root.get(i).get("error").asText());
                } else {
                    JsonNode result = root.get(i).get("result");
                    if (result.asText().startsWith("tensor")) {
                        writeTensorValueJson(g, Tensor.from(result.asText()));
                    } else {
                        writeDoubleValueJson(g, result.asDouble());
                    }
                }
                String steps = root.get(i).has("steps") ? root.get(i).get("steps").toString() : "";
                g.writeStringField("steps", steps);
            } catch (ParseException e) {
                g.writeStringField("error", "Unable to parse expression: '" + expression + "'");
            }

            g.writeEndObject();
        }
        g.writeEndArray();
        g.close();
        return out.toString();
    }

    private static void writeTensorValueJson(JsonGenerator g, Tensor tensor) throws IOException {
        g.writeStringField("type", tensor.type().toString());
        g.writeObjectFieldStart("value");
        g.writeStringField("literal", tensor.toString());
        g.writeArrayFieldStart("cells");
        for (Map.Entry<TensorAddress, Double> entry : tensor.cells().entrySet()) {
            TensorAddress address = entry.getKey();
            g.writeStartObject();
            g.writeObjectFieldStart("address");
            for (int i=0; i < address.size(); i++) {
                g.writeStringField(tensor.type().dimensions().get(i).name(), address.label(i));
            }
            g.writeEndObject();
            g.writeNumberField("value", entry.getValue());
            g.writeEndObject();
        }
        g.writeEndArray();
        g.writeEndObject();
    }

    private static void writeDoubleValueJson(JsonGenerator g, Double value) throws IOException {
        g.writeStringField("type", "double");
        g.writeNumberField("value", value);
    }

    private static String error(String msg) {
        return "{ \"error\": \"" + msg.replace("\"", "\\\"") + "\" }";
    }

    private static String toPrimitive(String expression) throws ParseException {
        return new RankingExpression(expression).toString();
    }

}
