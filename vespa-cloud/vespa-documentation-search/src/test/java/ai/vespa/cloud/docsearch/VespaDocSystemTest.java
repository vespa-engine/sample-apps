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
import java.util.Map;

import static java.net.http.HttpRequest.BodyPublishers.ofString;
import static org.junit.jupiter.api.Assertions.assertEquals;

@SystemTest
public class VespaDocSystemTest {

    Endpoint testEndpoint = TestRuntime.get().deploymentToTest().endpoint("default");
    ObjectMapper mapper = new ObjectMapper();

    @Test
    public void testSearchAndFeeding() throws Exception {
        feedTestDocs("/test-documents.json");

        HashSet<String> ids = getTestDocIDs();
        ids.forEach(id -> {System.out.println(id);});
        assertEquals(5, ids.size(), "test-documents.json has 5 documents");

        removeTestDocs(ids);
        ids = getTestDocIDs();
        assertEquals(0, ids.size(), "visit all docs and remove then");
    }

    @Test
    public void testInlinks() throws Exception {
        feedTestDocs("/test-documents.json");
        HashSet<String> ids = getTestDocIDs();
        assertEquals(5, ids.size());

        int numLinks = 0;
        String docWithOneInLink = getTestDoc("id:open:doc::open/documentation/access-logging.html");
        Iterator<JsonNode> inLinks = mapper.readTree(docWithOneInLink).get("fields").get("inlinks").iterator();
        while (inLinks.hasNext()) {
            inLinks.next();
            numLinks++;
        }
        assertEquals(1, numLinks);

        numLinks = 0;
        String docWithTwoInLinks = getTestDoc("id:open:doc::open/documentation/operations/admin-procedures.html");
        inLinks = mapper.readTree(docWithTwoInLinks).get("fields").get("inlinks").iterator();
        while (inLinks.hasNext()) {
            inLinks.next();
            numLinks++;
        }
        assertEquals(2, numLinks);
    }

    private void removeTestDocs(HashSet<String> ids) {
        System.out.println("** Start removing documents");
        ids.forEach(id -> {removeTestDoc(id);});
    }

    public void feedTestDocs(String testDocs) throws IOException {
        JsonNode docs = mapper.readTree(VespaDocSystemTest.class.getResourceAsStream(testDocs));
        Iterator<JsonNode> nodes = docs.elements();
        System.out.println("** Start feeding test documents");
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
        System.out.println("** Start dumping document IDs");
        do {
            String visitResult = visitDocuments(continuation.isEmpty() ? "" : "?continuation=" + continuation);
            continuation = getContinuationElement(visitResult);
            Iterator<JsonNode> documents = mapper.readTree(visitResult).path("documents").iterator();
            while (documents.hasNext()) {
                ids.add(documents.next().path("id").asText());
            }
        } while (!continuation.isEmpty());
        return ids;
    }

    public String getContinuationElement(String visitResult) throws Exception {
        return mapper.readTree(visitResult).path("continuation").asText();
    }

    public HttpResponse<String> feedTestDoc(JsonNode doc) {
        return testEndpoint.send(testEndpoint
                .request("/document/v1/open/doc/docid/" + "open" + doc.get("fields").get("path").textValue())
                .POST(ofString(doc.toString())));
    }

    private void removeTestDoc(String id) {
        testEndpoint.send(testEndpoint
                .request("/document/v1/open/doc/docid/" + id.split(":")[4]) // id:open:doc::open/documentation/annotations.html
                .DELETE());
    }

    public String getTestDoc(String id) {
        HttpResponse<String> res = testEndpoint.send(testEndpoint
                .request("/document/v1/open/doc/docid/" + id.split(":")[4]) // id:open:doc::open/documentation/annotations.html
                .GET());
        return res.body();
    }

    public String visitDocuments(String continuation) {
        HttpResponse<String> res = testEndpoint.send(testEndpoint
                .request("/document/v1/open/doc/docid/" + continuation)
                .GET());
        return res.body();
    }

    public void prettyPrint(String json) throws Exception {
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(new ObjectMapper().readTree(json)));
    }

    public int getNumSearchResults(String searchResult) throws Exception {
        return mapper.readTree(searchResult).get("root").get("fields").get("totalCount").asInt();
    }
}
