// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.reranker;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.ErrorMessage;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import com.yahoo.tensor.Tensor;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Converts a JSON result from a Vespa backend to Hits in a Result.
 *
 * @author bratseth
 */
class ResultReader {

    void read(String resultJson, Result result) {
        // Create ObjectMapper instance
        ObjectMapper objectMapper = new ObjectMapper();
        JsonFactory factory = new JsonFactory();

        try (JsonParser parser = factory.createParser(resultJson)) {
            // Read the tree structure from the JSON
            JsonNode jsonRoot = objectMapper.readTree(parser);
            JsonNode rootNode = jsonRoot.get("root");
            if (rootNode == null)
                throw new IllegalArgumentException("Expected a 'root' object in the JSON, got: " + jsonRoot);

            if (rootNode.get("fields") != null && rootNode.get("fields").get("totalCount") != null)
                result.setTotalHitCount(rootNode.get("fields").get("totalCount").asInt());

            if (rootNode.get("errors") != null)
                rootNode.get("errors").forEach(hit -> result.hits().addError(readError(hit)));
            if (rootNode.get("children") != null)
                rootNode.get("children").forEach(hit -> result.hits().add(readHit(hit, result.getQuery())));
        } catch (IOException e) {
            throw new IllegalArgumentException("Could not read result JSON", e);
        }
    }

    ErrorMessage readError(JsonNode errorObject) {
        return new ErrorMessage(errorObject.get("code").asInt(),
                                errorObject.get("summary").asText(),
                                errorObject.get("message") != null ? errorObject.get("message").asText() : null);
    }

    Hit readHit(JsonNode hitObject, Query query) {
        Hit hit = new Hit(hitObject.get("id").asText(), hitObject.get("relevance").asDouble(), query);
        // TODO: Source
        for (Iterator<Map.Entry<String, JsonNode>> i = hitObject.get("fields").fields(); i.hasNext(); ) {
            var fieldEntry = i.next();
            if ("matchfeatures".equals(fieldEntry.getKey()))
                hit.setField("matchfeatures", readFeatureData(fieldEntry.getValue()));
            if ("summaryfeatures".equals(fieldEntry.getKey()))
                hit.setField("summaryfeatures", readFeatureData(fieldEntry.getValue()));
            else
                hit.setField(fieldEntry.getKey(), toValue(fieldEntry.getValue()));
        }
        return hit;
    }

    FeatureData readFeatureData(JsonNode featureDataObject) {
        Map<String, Tensor> features = new HashMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> i = featureDataObject.fields(); i.hasNext(); ) {
            var fieldEntry = i.next();
            features.put(fieldEntry.getKey(), Tensor.from(fieldEntry.getValue().asDouble())); // TODO: Parse tensors
        }
        return new FeatureData(features);
    }

    public Object toValue(JsonNode fieldValue) {
        return switch (fieldValue.getNodeType()) {
            case NUMBER -> fieldValue.asDouble();
            case STRING -> fieldValue.asText();
            case BOOLEAN -> fieldValue.asBoolean();
            default -> fieldValue.asText();
        };
    }

}
