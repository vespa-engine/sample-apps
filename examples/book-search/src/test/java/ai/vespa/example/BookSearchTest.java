package ai.vespa.example;

import ai.vespa.testcontainers.VespaContainer;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class BookSearchTest {

    private static final ObjectMapper mapper = new ObjectMapper();
    private static VespaContainer vespa;
    private static BookClient client;

    private static final int CONVERGENCE_WAIT_MS = 500;

    @BeforeAll
    static void setUp() throws Exception {
        vespa = new VespaContainer().withApplicationPackage("app");
        vespa.start();

        client = new BookClient(vespa.getEndpoint());

        // Feed documents and wait for them to be indexed before running tests
        var result = client.feed(Path.of("data/documents.jsonl"));
        assertEquals(0, result.errors(), "Feed must complete without errors");
        Thread.sleep(CONVERGENCE_WAIT_MS);
    }

    @Test
    void textSearchFindsMatchingAuthor() throws Exception {
        var titles = titlesFrom(client.search("select * from book where userInput(@query)", "Tolkien"));
        assertTrue(titles.contains("The Lord of the Rings"),
                "Searching 'Tolkien' should return 'The Lord of the Rings'");
    }

    @Test
    void filterByYearRangeReturnsCorrectBooks() throws Exception {
        var response = client.search("select * from book where year >= 1950 and year <= 1960");
        var hits = mapper.readTree(response).path("root").path("children");
        assertFalse(hits.isEmpty(), "Year range filter should return results");
        for (var hit : hits) {
            int year = hit.path("fields").path("year").asInt();
            assertTrue(year >= 1950 && year <= 1960,
                    "All results must be within [1950, 1960], got: " + year);
        }
    }

    @Test
    void filterByThemeReturnsMatchingBooks() throws Exception {
        assertTrue(hitCount(client.search("select * from book where themes contains \"fantasy\"")) > 0,
                "Theme filter 'fantasy' should return results");
    }

    @Test
    void noResultsForImpossibleFilter() throws Exception {
        assertEquals(0, hitCount(client.search("select * from book where year < 1000")),
                "No books published before year 1000");
    }

    @Test
    void loanOutAndReturnBook() throws Exception {
        var docId = "the-lord-of-the-rings";

        client.setLoanedOut(docId, true);
        Thread.sleep(CONVERGENCE_WAIT_MS);
        assertTrue(titlesFrom(client.search("select * from book where loaned_out = true"))
                        .contains("The Lord of the Rings"),
                "Book should appear as loaned out after setLoanedOut(true)");

        client.setLoanedOut(docId, false);
        Thread.sleep(CONVERGENCE_WAIT_MS);
        assertTrue(titlesFrom(client.search("select * from book where loaned_out = false"))
                        .contains("The Lord of the Rings"),
                "Book should be available again after setLoanedOut(false)");
    }

    @AfterAll
    static void tearDown() {
        vespa.close();
    }

    // --- Helpers ---

    private static long hitCount(String json) throws Exception {
        return mapper.readTree(json).path("root").path("fields").path("totalCount").asLong();
    }

    private static List<String> titlesFrom(String json) throws Exception {
        var titles = new ArrayList<String>();
        for (var hit : mapper.readTree(json).path("root").path("children")) {
            var title = hit.path("fields").path("title").asText(null);
            if (title != null) titles.add(title);
        }
        return titles;
    }
}
