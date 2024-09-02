// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.reranker;

import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.result.FeatureData;
import com.yahoo.search.result.Hit;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * @author bratseth
 */
public class ResultReaderTest {

    @Test
    void testResultReader() {
        var reader = new ResultReader();
        var result = new Result(new Query());
        reader.read(RerankingTester.exampleResultJson, result);
        assertEquals(3, result.getTotalHitCount());
        assertEquals(3, result.hits().size());

        Hit hit1 = result.hits().get(0);
        assertEquals(0.3, hit1.getRelevance().getScore());
        assertEquals("id:mynamespace:music::love-id-here-to-stay", hit1.getId().toString());
        assertEquals("Diana Krall", hit1.fields().get("artist"));
        assertEquals("Love Is Here To Stay", hit1.fields().get("album"));
        assertEquals(2018.0, hit1.fields().get("year"));
        FeatureData featureData1 = (FeatureData)hit1.fields().get("summaryfeatures");
        assertEquals(3, featureData1.featureNames().size());
        assertEquals(1.0, featureData1.getDouble("fieldMatch(artist).completeness"));
        assertEquals(0.9, featureData1.getDouble("fieldMatch(artist).proximity"));

        Hit hit2 = result.hits().get(1);
        assertEquals(0.2, hit2.getRelevance().getScore());
        assertEquals("id:mynamespace:music::hardwired-to-self-destruct", hit2.getId().toString());
        assertEquals("Metallica", hit2.fields().get("artist"));
        assertEquals("Hardwired...To Self-Destruct", hit2.fields().get("album"));
        assertEquals(2016.0, hit2.fields().get("year"));
        FeatureData featureData2 = (FeatureData)hit2.fields().get("summaryfeatures");
        assertEquals(3, featureData2.featureNames().size());
        assertEquals(0.4, featureData2.getDouble("fieldMatch(artist).completeness"));
        assertEquals(0.3, featureData2.getDouble("fieldMatch(artist).proximity"));
    }

    @Test
    void testResultReaderOnError() {
        var reader = new ResultReader();
        var resultJson = """
{
    "root": {
        "id": "toplevel",
        "relevance": 1.0,
        "fields": {
            "totalCount": 0
        },
        "errors": [
            {
                "code": 4,
                "summary": "Invalid query parameter",
                "message": "Could not create query from YQL: Expected CALL or LITERAL, got READ_FIELD.",
                "stackTrace": "java.lang.IllegalArgumentException: Expected CALL or LITERAL, got READ_FIELD.\\n\\tat com.yahoo.search.yql.YqlParser.newUnexpectedArgumentException(YqlParser.java:1907)\\n\\tat com.yahoo.search.yql.YqlParser.instantiateLeafItem(YqlParser.java:1344)\\n\\tat com.yahoo.search.yql.YqlParser.buildTermSearch(YqlParser.java:1242)\\n\\tat com.yahoo.search.yql.YqlParser.convertExpression(YqlParser.java:353)\\n\\tat com.yahoo.search.yql.YqlParser.buildTree(YqlParser.java:296)\\n\\tat com.yahoo.search.yql.YqlParser.parse(YqlParser.java:275)\\n\\tat com.yahoo.search.yql.MinimalQueryInserter.insertQuery(MinimalQueryInserter.java:96)\\n\\tat com.yahoo.search.yql.MinimalQueryInserter.search(MinimalQueryInserter.java:81)\\n\\tat com.yahoo.search.Searcher.process(Searcher.java:134)\\n\\tat com.yahoo.processing.execution.Execution.process(Execution.java:112)\\n\\tat com.yahoo.search.searchchain.Execution.search(Execution.java:499)\\n\\tat com.yahoo.prelude.searcher.FieldCollapsingSearcher.search(FieldCollapsingSearcher.java:96)\\n\\tat com.yahoo.search.Searcher.process(Searcher.java:134)\\n\\tat com.yahoo.processing.execution.Execution.process(Execution.java:112)\\n\\tat com.yahoo.search.searchchain.Execution.search(Execution.java:499)\\n\\tat com.yahoo.prelude.querytransform.PhrasingSearcher.search(PhrasingSearcher.java:60)\\n\\tat com.yahoo.search.Searcher.process(Searcher.java:134)\\n\\tat com.yahoo.processing.execution.Execution.process(Execution.java:112)\\n\\tat com.yahoo.search.searchchain.Execution.search(Execution.java:499)\\n\\tat com.yahoo.prelude.statistics.StatisticsSearcher.search(StatisticsSearcher.java:235)\\n\\tat com.yahoo.search.Searcher.process(Searcher.java:134)\\n\\tat com.yahoo.processing.execution.Execution.process(Execution.java:112)\\n\\tat com.yahoo.search.searchchain.Execution.search(Execution.java:499)\\n\\tat com.yahoo.search.handler.SearchHandler.searchAndFill(SearchHandler.java:347)\\n\\tat com.yahoo.search.handler.SearchHandler.search(SearchHandler.java:392)\\n\\tat com.yahoo.search.handler.SearchHandler.handleBody(SearchHandler.java:268)\\n\\tat com.yahoo.search.handler.SearchHandler.handle(SearchHandler.java:177)\\n\\tat com.yahoo.container.jdisc.ThreadedHttpRequestHandler.handle(ThreadedHttpRequestHandler.java:77)\\n\\tat com.yahoo.container.jdisc.ThreadedHttpRequestHandler.handleRequest(ThreadedHttpRequestHandler.java:87)\\n\\tat com.yahoo.container.jdisc.ThreadedRequestHandler$RequestTask.processRequest(ThreadedRequestHandler.java:191)\\n\\tat com.yahoo.container.jdisc.ThreadedRequestHandler$RequestTask.run(ThreadedRequestHandler.java:185)\\n\\tat java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1136)\\n\\tat java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:635)\\n\\tat java.base/java.lang.Thread.run(Thread.java:840)\\n"
            }
        ]
    }
}""";
        var result = new Result(new Query());
        reader.read(resultJson, result);
        assertEquals(0, result.getTotalHitCount());
        assertEquals(1, result.hits().size());
        assertNotNull(result.hits().getError());
        assertEquals("Invalid query parameter", result.hits().getError().getMessage());
        assertEquals("Could not create query from YQL: Expected CALL or LITERAL, got READ_FIELD.",
                     result.hits().getError().getDetailedMessage());
        // (stack trace is not deserialized)
    }

}
