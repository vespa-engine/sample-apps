package com.mydomain.example;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static com.mydomain.example.SystemTestHttpUtils.postDocument;
import static com.mydomain.example.SystemTestHttpUtils.searchDocumentsGET;
import static com.mydomain.example.SystemTestHttpUtils.visitDocuments;

import static com.mydomain.example.SystemTestVespaUtils.getNumSearchResults;
import static com.mydomain.example.SystemTestVespaUtils.getTestEndpoint;
import static com.mydomain.example.SystemTestVespaUtils.prettyPrint;
import static com.mydomain.example.SystemTestVespaUtils.removeAllDocuments;
import static com.mydomain.example.SystemTestVespaUtils.getAllDocumentIds;

import static java.net.URLEncoder.encode;
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

    @BeforeEach
    public void init() throws Exception {
        removeAllDocuments("document/v1/music/music/docid/");
    }


    @Test  // Another annotation here so it will be run in pipeline
    public void simpleGETSearchTest() throws Exception {

        assertEquals(0, getNumSearchResults(searchDocumentsGET(getTestEndpoint()
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
        assertEquals(2, getNumSearchResults(searchResult), "Find all docs!");

        searchResult = searchDocumentsGET(getTestEndpoint()
                + "/search/?yql="
                + encode("select * from sources * where artist contains \"michael\";", StandardCharsets.UTF_8));
        prettyPrint(searchResult);
        assertEquals(1, getNumSearchResults(searchResult), "Find Michael Jackson!");
    }


    @Test
    public void simpleVisitTest() throws Exception {
        postDocument(getTestEndpoint() + "document/v1/music/music/docid/1", "/music-data-1.json");
        postDocument(getTestEndpoint() + "document/v1/music/music/docid/2", "/music-data-2.json");
        assertEquals(2, getAllDocumentIds("document/v1/music/music/docid/").size(), "Expected 2 documents after feeding");

        String visitResult = visitDocuments(getTestEndpoint() + "document/v1/music/music/docid/");
        prettyPrint(visitResult);
    }

}
