package ai.vespa.example.searchsuggest;

import ai.vespa.hosted.cd.Endpoint;
import ai.vespa.hosted.cd.SystemTest;
import ai.vespa.hosted.cd.TestRuntime;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestReporter;

import static org.junit.jupiter.api.Assertions.assertTrue;

@SystemTest
public class SearchSuggestSystemTest {

    @Test
    void testOutput(TestReporter testReporter){
        testReporter.publishEntry("I'm an empty test");
        assertTrue(true, "Text from assertion for comparison");
    }

}
