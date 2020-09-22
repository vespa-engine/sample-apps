package ai.vespa.cloud.docsearch;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.SystemTest;
import ai.vespa.hosted.cd.TestRuntime;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.http.HttpResponse;
import java.util.HashSet;
import java.util.Iterator;

import static java.net.http.HttpRequest.BodyPublishers.ofString;

@SystemTest
public class VespaDocSystemTest {

    Endpoint testEndpoint = TestRuntime.get().deploymentToTest().endpoint("default");

    @Test
    public void testSearchAndFeeding() throws Exception {
        feedTestDocs("/test-documents.json");

        HashSet<String> ids = getTestDocIDs();
        ids.forEach(id -> {System.out.println(id);});

        removeTestDocs(ids);
    }

    private void removeTestDocs(HashSet<String> ids) {
        System.out.println("** Start removing documents");
        ids.forEach(id -> {removeTestDoc(id);});
    }

    public void feedTestDocs(String testDocs) throws IOException {
        JsonNode docs = new ObjectMapper().readTree(VespaDocSystemTest.class.getResourceAsStream(testDocs));
        Iterator<JsonNode> nodes = docs.elements();
        System.out.println("** Starting feeding test documents");
        int i=0;
        while (nodes.hasNext()) {
            HttpResponse<String> res = feedTestDoc(nodes.next());
            assert(200 == res.statusCode());
            i++;
        }
        System.out.println("** Did feed " + i + " documents");
    }

    private HashSet<String> getTestDocIDs() throws Exception {
        HashSet<String> ids = new HashSet<>();
        String continuation = "";
        System.out.println("** Starting dumping document IDs");
        do {
            String visitResult = visitDocuments(continuation.isEmpty() ? "" : "?continuation=" + continuation);
            continuation = getContinuationElement(visitResult);
            Iterator<JsonNode> documents = new ObjectMapper().readTree(visitResult).path("documents").iterator();
            while (documents.hasNext()) {
                ids.add(documents.next().path("id").asText());
            }
        } while (!continuation.isEmpty());
        return ids;
    }

    public static String getContinuationElement(String visitResult) throws Exception {
        return new ObjectMapper().readTree(visitResult).path("continuation").asText();
    }

    public HttpResponse<String> feedTestDoc(JsonNode doc) {
        return testEndpoint.send(testEndpoint
                .request("/document/v1/open/doc/docid/" + doc.get("fields").get("path").textValue())
                .POST(ofString(doc.toString())));
    }

    private void removeTestDoc(String id) {
        testEndpoint.send(testEndpoint
                .request("/document/v1/open/doc/docid" + id.split(":")[4]) // id:open:doc::documentation/annotations.html
                .DELETE());
    }

    public String visitDocuments(String continuation) {
        HttpResponse<String> res = testEndpoint.send(testEndpoint
                .request("/document/v1/open/doc/docid/" + continuation)
                .GET());
        return res.body();
    }

    public static void prettyPrint(String json) throws Exception {
        System.out.println(new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(new ObjectMapper().readTree(json)));
    }

    public static int getNumSearchResults(String searchResult) throws Exception {
        return new ObjectMapper().readTree(searchResult).get("root").get("fields").get("totalCount").asInt();
    }
}
