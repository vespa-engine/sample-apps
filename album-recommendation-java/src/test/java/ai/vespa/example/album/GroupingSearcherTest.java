// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.example.album;

import com.yahoo.application.Application;
import com.yahoo.application.Networking;
import com.yahoo.application.container.Search;
import com.yahoo.component.ComponentSpecification;
import com.yahoo.component.chain.Chain;
import com.yahoo.search.Query;
import com.yahoo.search.Result;
import com.yahoo.search.Searcher;
import com.yahoo.search.searchchain.Execution;
import com.yahoo.search.yql.MinimalQueryInserter;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;

import static java.net.URLEncoder.encode;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class GroupingSearcherTest {


    @Test
    public void test_grouping() {
        try (Application app = Application.fromApplicationPackage(
                FileSystems.getDefault().getPath("src/main/application"),
                Networking.disable)) {
            String yql = encode("select * from sources * where true", StandardCharsets.UTF_8);
            Query query = new Query("/search/?yql=" + yql);

            Search search = app.getJDisc("default").search();
            Result result = search.process(ComponentSpecification.fromString("grouping"), query);
            System.out.println(result.getContext(false).getTrace());

            assertEquals("query 'TRUE'", result.getQuery().toString());
        }

    }

}
