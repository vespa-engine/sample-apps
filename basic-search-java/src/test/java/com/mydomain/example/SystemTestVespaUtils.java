package com.mydomain.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.HashSet;
import java.util.Iterator;

import static com.mydomain.example.SystemTestHttpUtils.removeDocument;
import static com.mydomain.example.SystemTestHttpUtils.visitDocuments;

public class SystemTestVespaUtils {

    public static String getTestEndpoint() {
        if (false) {   // something here when running in pipeline
            return "";
        }
        return "http://endpoint:10000/";
    }

    public static void removeAllDocuments(String path)  throws Exception {
        HashSet<String> ids = getAllDocumentIds(path);
        for (String id : ids) {
            removeDocument(getTestEndpoint() + path + id.split(":")[4]);  // id:music:music::2
        }
    }

    public static HashSet<String> getAllDocumentIds(String path) throws Exception {
        HashSet<String> ids = new HashSet<>();
        String continuation = "";
        do {
            String visitResult = visitDocuments(getTestEndpoint()
                    + path
                    + (continuation.isEmpty() ? "" : "?continuation=" + continuation));
            continuation = getContinuationElement(visitResult);
            Iterator<JsonNode> documents = new ObjectMapper().readTree(visitResult).path("documents").iterator();
            while (documents.hasNext()) {
                ids.add(documents.next().path("id").asText());
            }
        } while (!continuation.isEmpty());
        return ids;
    }

    public static int getNumSearchResults(String searchResult) throws Exception {
        return new ObjectMapper().readTree(searchResult).get("root").get("fields").get("totalCount").asInt();
    }

    public static String getContinuationElement(String visitResult) throws Exception {
        return new ObjectMapper().readTree(visitResult).path("continuation").asText();
    }

    public static void prettyPrint(String json) throws Exception {
        System.out.println(new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(new ObjectMapper().readTree(json)));
    }

}
