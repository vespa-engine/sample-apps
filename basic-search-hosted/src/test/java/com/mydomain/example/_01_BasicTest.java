package com.mydomain.example;

import ai.vespa.hosted.cd.StagingTest;
import ai.vespa.hosted.cd.SystemTest;
import com.yahoo.data.access.Inspector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;

import java.net.http.HttpRequest;
import java.util.Map;

import static com.mydomain.example.TestUtilities.assertStatusCode;
import static com.mydomain.example.TestUtilities.container;
import static com.mydomain.example.TestUtilities.entriesOf;
import static com.mydomain.example.TestUtilities.print;
import static com.mydomain.example.TestUtilities.requestFor;
import static com.mydomain.example.TestUtilities.send;
import static com.mydomain.example.TestUtilities.toInspector;
import static java.net.http.HttpRequest.BodyPublishers.noBody;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * <p>
 *     This test verifies the absolute basics of the Vespa installation. Please read the tests in order.
 * </p>
 *
 * @author jonmv
 */
@SystemTest
@StagingTest
@Tag("system")
@Tag("staging")
@DisplayName("Test that the application")
class _01_BasicTest {

    /**
     * The {@code /status.html} path indicates whether the container is up and healthy,
     * and is used by the routing layer to determine which containers to route traffic to.
     */
    @Test
    @DisplayName("serves 200 OK on /status.html")
    void isUp() {
        var response = send(container(), requestFor(container(), "/status.html"));
        assertStatusCode(200, response);
    }

    /**
     * The {@code /ApplicationStatus} path serves useful information about the application.
     * This test verifies some of this information is as expected from the deployed {@code services.xml}.
     */
    @Test
    @DisplayName("has expected components, as per services.xml")
    void hasComponents() {
        var response = send(container(), requestFor(container(), "/ApplicationStatus"));
        assertStatusCode(200, response);

        Inspector root = toInspector(response.body());
        print(root);

        assertTrue(entriesOf(root.field("handlers"))
                           .flatMap(handler -> entriesOf(handler.field("serverBindings")))
                           .anyMatch(inspector -> inspector.asString().endsWith("/search/*")),
                   "The <search /> element in services.xml should create a search handler");

        assertTrue(entriesOf(root.field("handlers"))
                           .flatMap(handler -> entriesOf(handler.field("serverBindings")))
                           .anyMatch(inspector -> inspector.asString().endsWith("/document/v1/*")),
                   "The <document-api /> element in services.xml should create a search handler");

        assertTrue(root.field("searchChains").field("books").valid(),
                   "Each content cluster should have its own named search chain");
    }


    /**
     * The search API at {@code /search/} allows querying of the Vespa instance.
     *
     * Parameters for the search are passed as URL properties, or as the request body, as per the
     * <a href="https://docs.vespa.ai/documentation/search-api.html">documentation</a>.
     */
    @Nested
    @DisplayName("search endpoint")
    class SearchApi {

        @Test
        @DisplayName("rejects empty queries")
        void rejectsEmptyQueries() {
            var response = send(container(), requestFor(container(), "/search/"));
            assertStatusCode(400, response);

            Inspector root = toInspector(response.body());
            print(root);

            assertEquals("No query", root.field("root").field("errors").entry(0).field("message").asString());
        }

        @Test
        @DisplayName("returns an empty result when there are no documents")
        void emptyResult() {
            var yql = "SELECT * FROM SOURCES books WHERE default CONTAINS \"Shakespeare\";";
            var response = send(container(), requestFor(container(), "/search/", Map.of("yql", yql)));
            assertStatusCode(200, response);

            Inspector root = toInspector(response.body());
            print(root);

            assertEquals(0, root.field("root").field("fields").field("totalCount").asLong());
            assertFalse(root.field("root").field("hits").valid());
        }

    }


    /**
     * The document api at {@code /document/v1/} allows CRUD operations on documents.
     *
     * Documents are in JSON format, as specified in the
     * <a href="https://docs.vespa.ai/documentation/reference/document-json-format.html">documentation</a>.
     *
     * Each document <a href="https://docs.vespa.ai/documentation/documents.html#document-ids">has an id</a>,
     * and requests to the document API encode this in the URL path.
     *
     * These tests are stateful, so methods have to be run in a particular order.
     * (Well, really, they could all be one test, but this shows test method order, which may be useful for larger tests.)
     */
    @Nested
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    @DisplayName("document endpoint")
    class DocumentApi {

        String namespace = "books";        // The namespace of the document — irrelevant in this example
        String documentType = "person";    // The document type of the document to create — see person.sd
        String userSpecified = "hamlet";   // The "name" of this particular document
        String documentApiPath = "/document/v1/" + namespace + "/" + documentType + "/docid/" + userSpecified;

        @Test
        @Order(1)
        @DisplayName("creates a person document")
        void createDocument() {
            String document = "{\n" +
                              "    \"fields\": {\n" +
                              "        \"name\": \"Hamlet\",\n" +  // String name
                              "        \"birth_year\": 1572\n" +   // Integer birth year
                              "    }\n" +
                              "}\n";

            var body = HttpRequest.BodyPublishers.ofString(document, UTF_8);
            var response = send(container(), requestFor(container(), documentApiPath).method("POST", body));
            assertStatusCode(200, response);

            Inspector root = toInspector(response.body());
            print(root);
            assertEquals("id:books:person::hamlet", root.field("id").asString(),
                         "Created document should have the expected id");
        }

        @Test
        @Order(2)
        @DisplayName("updates an existing document")
        void updateDocument() {
            String update = "{\n" +
                            "    \"fields\": {\n" +
                            "        \"birth_year\": {\n" +     // Update this field ...
                            "            \"assign\": 1573\n" +  // ... by assigning a new value
                            "        }\n" +
                            "    }\n" +
                            "}\n";

            var body = HttpRequest.BodyPublishers.ofString(update, UTF_8);
            var response = send(container(), requestFor(container(), documentApiPath).method("PUT", body));
            assertStatusCode(200, response);

            Inspector root = toInspector(response.body());
            print(root);
            assertEquals("id:books:person::hamlet", root.field("id").asString());
        }

        @Test
        @Order(3)
        @DisplayName("reads an existing document")
        void getDocument() {
            var response = send(container(), requestFor(container(), documentApiPath));
            assertStatusCode(200, response);

            Inspector root = toInspector(response.body());
            print(root);
            assertEquals("id:books:person::hamlet", root.field("id").asString());
            assertEquals("Hamlet", root.field("fields").field("name").asString());
            assertEquals(1573, root.field("fields").field("birth_year").asLong());
        }

        @Test
        @Order(4)
        @DisplayName("deletes an existing document")
        void removeDocument() {
            var response = send(container(), requestFor(container(), documentApiPath).method("DELETE", noBody()));
            assertStatusCode(200, response);

            Inspector root = toInspector(response.body());
            print(root);
            assertEquals("id:books:person::hamlet", root.field("id").asString());
        }

    }

}
