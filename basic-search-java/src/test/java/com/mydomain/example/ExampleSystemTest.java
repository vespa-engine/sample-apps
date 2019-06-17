package com.mydomain.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Iterator;

import static java.net.URLEncoder.encode;
import static java.net.http.HttpRequest.BodyPublishers.ofInputStream;
import static org.junit.jupiter.api.Assertions.assertEquals;


/**
 * System tests - demonstrates:
 * <ol>
 *     <li>How to set up the endpoint to the test instance</li>
 *     <li>Feed to the test instance</li>
 *     <li>Query the test instance</li>
 *     <li>Visit the test instance</li>
 * </ol>
 */

@Tag("SystemTest")
public class ExampleSystemTest {

    private String getTestEndpoint() {
        if (false) {   // something here when running in pipeline
            return "";
        }
        return "http://endpoint:10000/";
    }


    @BeforeEach
    public void init() throws Exception {
        removeAllDocuments("document/v1/music/music/docid/");
    }


    @Test  // Another annotation here so it will be run in pipeline
    public void simpleGETSearchTest() throws Exception {

        assertEquals(0, getSearchResults(searchDocumentsGET(getTestEndpoint()
                + "/search/?yql="
                + encode("select * from sources * where sddocname contains \"music\";", StandardCharsets.UTF_8))),
                "Expected 0 documents before feeding");

        postDocument(getTestEndpoint() + "document/v1/music/music/docid/1", "/music-data-1.json");
        postDocument(getTestEndpoint() + "document/v1/music/music/docid/2", "/music-data-2.json");
        assertEquals(2, getAllDocumentIds("document/v1/music/music/docid/").size(), "Expected 2 documents after feeding");

        String searchResult = searchDocumentsGET(getTestEndpoint()
                + "/search/?yql="
                + encode("select * from sources * where title contains \"bad\";", StandardCharsets.UTF_8));
        prettyPrint(searchResult);
        assertEquals(2, getSearchResults(searchResult), "Find all docs!");

        searchResult = searchDocumentsGET(getTestEndpoint()
                + "/search/?yql="
                + encode("select * from sources * where artist contains \"michael\";", StandardCharsets.UTF_8));
        prettyPrint(searchResult);
        assertEquals(1, getSearchResults(searchResult), "Find Michael Jackson!");
    }


    @Test
    public void simpleVisitTest() throws Exception {
        postDocument(getTestEndpoint() + "document/v1/music/music/docid/1", "/music-data-1.json");
        postDocument(getTestEndpoint() + "document/v1/music/music/docid/2", "/music-data-2.json");
        assertEquals(2, getAllDocumentIds("document/v1/music/music/docid/").size(), "Expected 2 documents after feeding");

        String visitResult = visitDocuments(getTestEndpoint() + "document/v1/music/music/docid/");
        prettyPrint(visitResult);
    }


    public void removeAllDocuments(String path)  throws Exception {
        HashSet<String> ids = getAllDocumentIds(path);
        for (String id : ids) {
            removeDocument(getTestEndpoint() + path + id.split(":")[4]);  // id:music:music::2
        }
    }

    public HashSet<String> getAllDocumentIds(String path) throws Exception {
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

    public int getSearchResults(String searchResult) throws Exception {
        return new ObjectMapper().readTree(searchResult).get("root").get("fields").get("totalCount").asInt();
    }

    public String getContinuationElement(String visitResult) throws Exception {
        return new ObjectMapper().readTree(visitResult).path("continuation").asText();
    }

    public void prettyPrint(String json) throws Exception {
        System.out.println(new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(new ObjectMapper().readTree(json)));
    }

    public boolean postDocument(String docApiPut, String feed) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(docApiPut))
                .header("Content-Type", "application/json")
                .POST(ofInputStream(() -> getClass().getResourceAsStream(feed)))
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
    }

    public String visitDocuments(String docApiVisit) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(docApiVisit))
                .GET()
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString()).body();
    }

    public String searchDocumentsGET(String searchApiQuery) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(searchApiQuery))
                .GET()
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString()).body();
    }

    public boolean removeDocument(String deleteRequest) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(deleteRequest))
                .DELETE()
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
    }

}
