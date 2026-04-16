package ai.vespa.example;

import ai.vespa.testcontainers.VespaContainer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class BookSearchTest {
    private static final int    POLL_TIMEOUT_MS = 5_000;
    private static final ObjectMapper mapper = new ObjectMapper();
    private static VespaContainer vespa;
    private static BookClient bookClient;

    @BeforeAll
    static void setUp() throws Exception {
        vespa = new VespaContainer().withApplicationPackage("app");
        vespa.start();

        bookClient = new BookClient(vespa.getEndpoint());

        BookClient.FeedResult result = bookClient.feed(Path.of("data/documents.jsonl"));
        assertEquals(0, result.errors(), "Feed must complete without errors");
        awaitCondition("20 documents indexed",
                () -> hitCount(bookClient.query("select * from book where true")) == 20);
    }

    @Test
    void textSearchFindsMatchingAuthor() throws Exception {
        List<String> titles = titlesFrom(bookClient.query("select * from book where userInput(@query)", "Tolkien"));
        assertTrue(titles.contains("The Lord of the Rings"),
                "Searching 'Tolkien' should return 'The Lord of the Rings'");
    }

    @Test
    void filterByYearRangeReturnsCorrectBooks() throws Exception {
        String response = bookClient.query("select * from book where year >= 1950 and year <= 1960");
        JsonNode hits = mapper.readTree(response).path("root").path("children");
        assertFalse(hits.isEmpty(), "Year range filter should return results");
        for (JsonNode hit : hits) {
            int year = hit.path("fields").path("year").asInt();
            assertTrue(year >= 1950 && year <= 1960,
                    "All results must be within [1950, 1960], got: " + year);
        }
    }

    @Test
    void filterByThemeReturnsMatchingBooks() throws Exception {
        assertTrue(hitCount(bookClient.query("select * from book where themes contains \"fantasy\"")) > 0,
                "Theme filter 'fantasy' should return results");
    }

    @Test
    void noResultsForImpossibleFilter() throws Exception {
        assertEquals(0, hitCount(bookClient.query("select * from book where year < 1000")),
                "No books published before year 1000");
    }

    @Test
    void loanOutAndReturnBook() throws Exception {
        String docId = "the-lord-of-the-rings";

        bookClient.setLoanedOut(docId, true);
        awaitCondition("'The Lord of the Rings' loaned_out=true",
                () -> titlesFrom(bookClient.query("select * from book where loaned_out = true"))
                        .contains("The Lord of the Rings"));

        bookClient.setLoanedOut(docId, false);
        awaitCondition("'The Lord of the Rings' loaned_out=false",
                () -> titlesFrom(bookClient.query("select * from book where loaned_out = false"))
                        .contains("The Lord of the Rings"));
    }

    @AfterAll
    static void tearDown() throws Exception {
        vespa.close();
    }

    // --- Helpers ---

    private static void awaitCondition(String description, Callable<Boolean> condition) throws Exception {
        for (int i = 0; i < POLL_TIMEOUT_MS / 100; i++) {
            if (condition.call()) return;
            Thread.sleep(100);
        }
        fail("Condition not met within " + POLL_TIMEOUT_MS + "ms: " + description);
    }

    private static long hitCount(String json) throws Exception {
        return mapper.readTree(json).path("root").path("fields").path("totalCount").asLong();
    }

    private static List<String> titlesFrom(String json) throws Exception {
        List<String> titles = new ArrayList<>();
        for (JsonNode hit : mapper.readTree(json).path("root").path("children")) {
            String title = hit.path("fields").path("title").asText(null);
            if (title != null) titles.add(title);
        }
        return titles;
    }
}
